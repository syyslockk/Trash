"""Относительный индекс риска (0..1) по упрощённой модели Кесслера. Это не вероятность столкновения.

Частота столкновений ~ плотность мусора x сечение x скорость сближения.
Сетка - высота x наклонение. В каждой клетке:
    hazard   - насколько опасно лететь в этой клетке (от обломков всех наклонений
               на той же высоте, с учётом скорости сближения);
    exposure - hazard x число целей в клетке: где риск для тех, кто там летает.
Обе величины делятся на свой максимум, поэтому они относительные.

Допущения: орбиты круговые, плотность усредняется по слою высоты без учёта
широты, сечение одинаковое для всех, в каталоге только объекты крупнее ~10 см.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from source import config

log = logging.getLogger(__name__)


@dataclass
class Grid:
    alt_edges: np.ndarray
    inc_edges: np.ndarray

    @property
    def alt_centers(self) -> np.ndarray:
        return (self.alt_edges[:-1] + self.alt_edges[1:]) / 2

    @property
    def inc_centers(self) -> np.ndarray:
        return (self.inc_edges[:-1] + self.inc_edges[1:]) / 2


@dataclass
class RiskResult:
    grid: pd.DataFrame       # одна строка - одна клетка «высота x наклонение»
    profile: pd.DataFrame    # одна строка - один слой высоты
    hazard: np.ndarray       # hazard_rel, форма (n_alt, n_inc)
    exposure: np.ndarray     # exposure_rel, форма (n_alt, n_inc)
    cell: Grid


def make_grid() -> Grid:
    return Grid(
        alt_edges=np.arange(config.ALT_MIN, config.ALT_MAX + config.ALT_STEP, config.ALT_STEP, dtype=float),
        inc_edges=np.arange(config.INC_MIN, config.INC_MAX + config.INC_STEP, config.INC_STEP, dtype=float),
    )


def vrel_factor_matrix(inc_deg: np.ndarray, n_steps: int = config.N_NODE_STEPS) -> np.ndarray:
    """F[i1, i2] - средний множитель 2*sin(D/2) для пары наклонений.

    Скорость сближения двух круговых орбит одной высоты: v_rel = 2 v_circ sin(D/2),
    где cos D = cos i1 cos i2 + sin i1 sin i2 cos(dRAAN). Усредняем по dRAAN в [0, 2pi).
    """
    i = np.radians(np.asarray(inc_deg, dtype=float))
    i1, i2 = i[:, None, None], i[None, :, None]
    d_raan = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)[None, None, :]
    cos_d = np.clip(np.cos(i1) * np.cos(i2) + np.sin(i1) * np.sin(i2) * np.cos(d_raan), -1, 1)
    return np.sqrt(2.0 * (1.0 - cos_d)).mean(axis=2)


def is_target(df: pd.DataFrame) -> pd.Series:
    """Цели риска: действующие спутники и ракетные ступени."""
    return ((df["object_type"] == "PAY") & df["is_active"].astype(bool)) | (df["object_type"] == "R/B")


def risk_inputs(objects: pd.DataFrame) -> pd.DataFrame:
    """Объекты, которые идут в расчёт риска, с флагами is_target и is_debris."""
    df = objects[objects["eccentricity"] <= config.RISK_MAX_ECCENTRICITY].copy()
    df["is_debris"] = df["object_type"] == "DEB"
    df["is_target"] = is_target(df)
    log.info("Риск: %d объектов (без %d сильно вытянутых орбит, e > %.2f): целей %d, обломков %d",
             len(df), len(objects) - len(df), config.RISK_MAX_ECCENTRICITY,
             int(df["is_target"].sum()), int(df["is_debris"].sum()))
    return df


def _histogram(df: pd.DataFrame, cell: Grid) -> np.ndarray:
    h, _, _ = np.histogram2d(df["mean_alt_km"], df["inclination_deg"],
                             bins=[cell.alt_edges, cell.inc_edges])
    return h


def _normalize(x: np.ndarray) -> np.ndarray:
    peak = x.max()
    return x / peak if peak > 0 else x


def compute_risk(objects: pd.DataFrame) -> RiskResult:
    cell = make_grid()
    alt_c, inc_c = cell.alt_centers, cell.inc_centers
    df = risk_inputs(objects)

    n_debris = _histogram(df[df["is_debris"]], cell)
    n_payload = _histogram(df[df["is_target"] & (df["object_type"] == "PAY")], cell)
    n_rocket = _histogram(df[df["is_target"] & (df["object_type"] == "R/B")], cell)
    n_targets = n_payload + n_rocket

    v_circ = np.sqrt(config.GM_EARTH / (config.R_EARTH + alt_c))           # км/с
    volume = 4 * np.pi * (config.R_EARTH + alt_c) ** 2 * config.ALT_STEP   # км^3
    dens_deb = gaussian_filter(n_debris, sigma=config.SMOOTH_SIGMA, mode="constant") / volume[:, None]

    F = vrel_factor_matrix(inc_c)
    hazard = (dens_deb @ F.T) * v_circ[:, None] * config.SIGMA_KM2
    exposure = hazard * n_targets
    hazard_rel, exposure_rel = _normalize(hazard), _normalize(exposure)

    aa, ii = np.meshgrid(alt_c, inc_c, indexing="ij")
    grid = pd.DataFrame({
        "alt_center_km": aa.ravel(),
        "inc_center_deg": ii.ravel(),
        "n_targets": n_targets.ravel().astype(int),
        "n_active_satellites": n_payload.ravel().astype(int),
        "n_rocket_bodies": n_rocket.ravel().astype(int),
        "n_debris": n_debris.ravel().astype(int),
        "hazard_rel": hazard_rel.ravel(),
        "exposure_rel": exposure_rel.ravel(),
    })

    layer = exposure_rel.sum(axis=1)
    profile = pd.DataFrame({
        "alt_center_km": alt_c,
        "n_targets": n_targets.sum(axis=1).astype(int),
        "n_debris": n_debris.sum(axis=1).astype(int),
        "debris_density_rel": _normalize(dens_deb.sum(axis=1)),
        "exposure_rel": _normalize(layer),
    })
    return RiskResult(grid=grid, profile=profile, hazard=hazard_rel, exposure=exposure_rel, cell=cell)


def object_hazard(objects: pd.DataFrame, risk: RiskResult) -> np.ndarray:
    """hazard_rel в точке орбиты каждого объекта (линейная интерполяция по центрам клеток)."""
    alt_c, inc_c = risk.cell.alt_centers, risk.cell.inc_centers
    interp = RegularGridInterpolator((alt_c, inc_c), risk.hazard, method="linear")
    pts = np.column_stack([
        np.clip(objects["mean_alt_km"].to_numpy(dtype=float), alt_c[0], alt_c[-1]),
        np.clip(objects["inclination_deg"].to_numpy(dtype=float), inc_c[0], inc_c[-1]),
    ])
    return interp(pts)


def _range(center: pd.Series, step: float) -> pd.Series:
    return center.map(lambda c: f"{c - step / 2:.0f}-{c + step / 2:.0f}")


def top_zones(risk: RiskResult, n: int = config.TOP_N) -> pd.DataFrame:
    """Клетки с наибольшим exposure_rel: где под угрозой больше всего целей."""
    top = risk.grid[risk.grid["exposure_rel"] > 0].nlargest(n, "exposure_rel")
    return _zone_table(top)


def top_hazard_zones(
    risk: RiskResult,
    n: int = config.TOP_N,
    min_targets: int = config.HAZARD_ZONE_MIN_TARGETS,
) -> pd.DataFrame:
    """Клетки с наибольшим hazard_rel среди тех, где летают цели: где опаснее всего лететь.

    В отличие от top_zones не зависит от того, сколько там спутников.
    """
    grid = risk.grid
    top = grid[(grid["n_targets"] >= min_targets) & (grid["hazard_rel"] > 0)].nlargest(n, "hazard_rel")
    return _zone_table(top)


def _zone_table(top: pd.DataFrame) -> pd.DataFrame:
    top = top.assign(
        alt_range_km=_range(top["alt_center_km"], config.ALT_STEP),
        inc_range_deg=_range(top["inc_center_deg"], config.INC_STEP),
    )
    cols = ["alt_range_km", "inc_range_deg", "n_targets", "n_active_satellites",
            "n_rocket_bodies", "n_debris", "hazard_rel", "exposure_rel"]
    return top[cols].reset_index(drop=True)


def top_objects(clustered: pd.DataFrame, n: int = config.TOP_N) -> pd.DataFrame:
    """Цели (действующие спутники и ступени) с наибольшим hazard_rel на их орбите."""
    in_model = clustered["eccentricity"] <= config.RISK_MAX_ECCENTRICITY
    targets = clustered[in_model & is_target(clustered)]
    top = targets.sort_values(["hazard_rel", "norad_id"], ascending=[False, True]).head(n)
    cols = ["norad_id", "object_name", "object_type", "category", "mean_alt_km",
            "inclination_deg", "cluster", "hazard_rel"]
    return top[cols].reset_index(drop=True)


def find_risk_bands(profile: pd.DataFrame) -> List[dict]:
    """Непрерывные диапазоны высот, где exposure_rel слоя выше порога (оболочки на 3D)."""
    values = profile["exposure_rel"].to_numpy()
    alt_c = profile["alt_center_km"].to_numpy()
    idx = np.flatnonzero(values >= config.SHELL_THRESHOLD)
    if len(idx) == 0:
        return []
    bands = []
    for grp in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        w = values[grp]
        bands.append({
            "centre_km": round(float(np.average(alt_c[grp], weights=w)), 1),
            "lo_km": float(alt_c[grp[0]] - config.ALT_STEP / 2),
            "hi_km": float(alt_c[grp[-1]] + config.ALT_STEP / 2),
            "peak": round(float(w.max()), 4),
        })
    return sorted(bands, key=lambda b: b["peak"], reverse=True)[:config.MAX_SHELLS]
