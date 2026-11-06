"""Объединение SATCAT и GP по NORAD_CAT_ID, расчёт высот, фильтр LEO, категории.

Основа набора - SATCAT: в нём есть все объекты на орбите и их тип
(PAY / R/B / DEB / UNK). Для объектов из GP-групп высоты пересчитываются
по элементам орбиты (они свежее), и добавляется RAAN. У остальных объектов
высоты берутся из SATCAT, а RAAN и эпоха остаются пустыми.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from source import config
from source.loader.celestrak import read_gp, read_satcat

log = logging.getLogger(__name__)

OBJECT_COLUMNS = [
    "norad_id", "object_name", "object_type", "category", "is_active",
    "perigee_km", "apogee_km", "mean_alt_km", "inclination_deg",
    "eccentricity", "raan_deg", "epoch", "source",
]

# Категория по типу объекта, если он не из облака обломков
TYPE_CATEGORY = {
    "R/B": "rocket_body",
    "DEB": "debris",
    "UNK": "unknown",
}


class ProcessingError(RuntimeError):
    """Нет входных данных или они не подходят."""


def kepler_heights(mean_motion_rev_day, eccentricity) -> pd.DataFrame:
    """Перигей, апогей и средняя высота над Землёй (км) по 3-му закону Кеплера.

    n [об/сут] -> рад/с; a = (GM / n^2)^(1/3).
    """
    n = np.asarray(mean_motion_rev_day, dtype=float) * 2 * np.pi / config.SECONDS_PER_DAY
    e = np.asarray(eccentricity, dtype=float)
    a = (config.GM_EARTH / n ** 2) ** (1 / 3)
    perigee = a * (1 - e) - config.R_EARTH
    apogee = a * (1 + e) - config.R_EARTH
    return pd.DataFrame({
        "perigee_km": perigee,
        "apogee_km": apogee,
        "mean_alt_km": (perigee + apogee) / 2,
    })


def satcat_on_orbit(satcat: pd.DataFrame) -> pd.DataFrame:
    """Объекты SATCAT на орбите Земли с заполненными орбитальными полями."""
    on_orbit = (
        satcat["DECAY_DATE"].isna()
        & (satcat["ORBIT_CENTER"] == "EA")
        & (satcat["ORBIT_TYPE"] == "ORB")
    )
    df = satcat[on_orbit]
    has_orbit = df[["INCLINATION", "APOGEE", "PERIGEE"]].notna().all(axis=1)
    log.info("SATCAT: %d объектов на орбите, %d без орбитальных данных отброшено",
             len(df), int((~has_orbit).sum()))
    df = df[has_orbit]

    r_apo = config.R_EARTH + df["APOGEE"]
    r_per = config.R_EARTH + df["PERIGEE"]
    return pd.DataFrame({
        "norad_id": df["NORAD_CAT_ID"].astype(int),
        "object_name": df["OBJECT_NAME"],
        "object_type": df["OBJECT_TYPE"],
        "is_active": df["OPS_STATUS_CODE"].isin(config.ACTIVE_OPS_CODES),
        "perigee_km": df["PERIGEE"].astype(float),
        "apogee_km": df["APOGEE"].astype(float),
        "mean_alt_km": (df["APOGEE"] + df["PERIGEE"]) / 2,
        "inclination_deg": df["INCLINATION"].astype(float),
        "eccentricity": (r_apo - r_per) / (r_apo + r_per),
    }).reset_index(drop=True)


def gp_elements(gp_groups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Склеивает GP-группы в одну таблицу с высотами по Кеплеру и колонкой group."""
    frames = []
    for group, gp in gp_groups.items():
        if gp.empty:
            continue
        heights = kepler_heights(gp["MEAN_MOTION"], gp["ECCENTRICITY"])
        frames.append(pd.DataFrame({
            "norad_id": gp["NORAD_CAT_ID"].astype(int).to_numpy(),
            "gp_name": gp["OBJECT_NAME"].to_numpy(),
            "group": group,
            "gp_perigee_km": heights["perigee_km"].to_numpy(),
            "gp_apogee_km": heights["apogee_km"].to_numpy(),
            "gp_mean_alt_km": heights["mean_alt_km"].to_numpy(),
            "gp_inclination_deg": gp["INCLINATION"].astype(float).to_numpy(),
            "gp_eccentricity": gp["ECCENTRICITY"].astype(float).to_numpy(),
            "raan_deg": gp["RA_OF_ASC_NODE"].astype(float).to_numpy(),
            "epoch": gp["EPOCH"].to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=["norad_id", "group"])
    gp = pd.concat(frames, ignore_index=True)
    # Объект может быть в нескольких группах: облако важнее, чем active
    gp["_is_cloud"] = gp["group"] != config.GP_ACTIVE_GROUP
    gp = (gp.sort_values(["norad_id", "_is_cloud"], ascending=[True, False])
            .drop_duplicates("norad_id")
            .drop(columns="_is_cloud"))
    return gp.reset_index(drop=True)


def _category(df: pd.DataFrame) -> pd.Series:
    cloud = df["group"].map(
        {g: c for g, c in config.GP_GROUPS.items() if g != config.GP_ACTIVE_GROUP})
    by_type = df["object_type"].map(TYPE_CATEGORY)
    payload = np.where(df["is_active"], "active_satellite", "inactive_payload")
    by_type = by_type.where(df["object_type"] != "PAY", payload)
    return cloud.fillna(by_type).fillna("unknown")


def build_objects(satcat: pd.DataFrame, gp_groups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Все объекты на орбите Земли в формате OBJECT_COLUMNS (без фильтра по высоте)."""
    base = satcat_on_orbit(satcat)
    gp = gp_elements(gp_groups)
    df = base.merge(gp, on="norad_id", how="outer", indicator=True)

    only_gp = df["_merge"] == "right_only"
    if only_gp.any():
        # Например, объект ещё не попал в SATCAT или уже помечен там как сошедший
        log.info("GP: %d объектов нет среди SATCAT на орбите, тип взят по группе",
                 int(only_gp.sum()))
        df.loc[only_gp, "object_name"] = df.loc[only_gp, "gp_name"]
        df.loc[only_gp, "object_type"] = np.where(
            df.loc[only_gp, "group"] == config.GP_ACTIVE_GROUP, "PAY", "DEB")
        df.loc[only_gp, "is_active"] = False

    in_gp = df["_merge"] != "left_only"
    _log_height_check(df[df["_merge"] == "both"])
    for col in ["perigee_km", "apogee_km", "mean_alt_km", "inclination_deg", "eccentricity"]:
        df.loc[in_gp, col] = df.loc[in_gp, "gp_" + col]

    # Спутник из группы active считаем действующим, даже если в SATCAT другой код.
    # В группу иногда попадают ступени свежих запусков - их не трогаем.
    in_active = (df["group"] == config.GP_ACTIVE_GROUP) & (df["object_type"] == "PAY")
    df["is_active"] = df["is_active"].astype(bool) | in_active
    df["source"] = np.where(in_gp, "gp", "satcat")
    df["category"] = _category(df)

    df = df[OBJECT_COLUMNS].dropna(subset=["mean_alt_km", "inclination_deg"])
    return df.sort_values("norad_id").reset_index(drop=True)


def _log_height_check(both: pd.DataFrame) -> None:
    """Сверка высот по Кеплеру с APOGEE/PERIGEE из SATCAT (для контроля, в лог)."""
    if both.empty:
        return
    diff = (both["gp_mean_alt_km"] - both["mean_alt_km"]).abs()
    log.info("Высоты GP и SATCAT: медиана расхождения %.1f км, у %.1f%% объектов больше 20 км",
             diff.median(), 100 * (diff > 20).mean())


def filter_leo(df: pd.DataFrame) -> pd.DataFrame:
    leo = df["mean_alt_km"].between(config.ALT_MIN, config.ALT_MAX)
    return df[leo].reset_index(drop=True)


def process(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Читает data/raw и возвращает объекты на низких орбитах."""
    raw_dir = Path(raw_dir or config.RAW_DIR)
    if not (raw_dir / config.SATCAT_FILE).exists():
        raise ProcessingError(
            f"Нет {raw_dir / config.SATCAT_FILE}. Сначала запустите: python -m source.loader")

    gp_groups = {}
    for group in config.GP_GROUPS:
        if (raw_dir / config.gp_filename(group)).exists():
            gp_groups[group] = read_gp(group, raw_dir)
        else:
            log.warning("Нет файла GP-группы %s, она пропущена", group)

    objects = build_objects(read_satcat(raw_dir), gp_groups)
    leo = filter_leo(objects)
    log.info("Объектов на орбите: %d, из них на низких орбитах (%d-%d км): %d",
             len(objects), config.ALT_MIN, config.ALT_MAX, len(leo))
    for category, count in leo["category"].value_counts().items():
        log.info("  %-22s %6d", category, count)
    return leo


def save(objects: pd.DataFrame, processed_dir: Optional[Path] = None) -> Path:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / config.OBJECTS_FILE
    objects.to_csv(path, index=False)
    log.info("Сохранено: %s", path)
    return path


def read_objects(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_csv(Path(processed_dir or config.PROCESSED_DIR) / config.OBJECTS_FILE,
                       keep_default_na=False, na_values=[""])
