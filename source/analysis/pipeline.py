"""Шаг analysis целиком: кластеризация -> проверка по облакам -> риск -> таблицы.

run() только считает и возвращает результат, save() пишет файлы в data/processed.
Так GUI и CLI используют один и тот же код.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from source import config
from source.analysis import clustering, risk, validation

log = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    clustered: pd.DataFrame     # objects + cluster + hazard_rel
    clusters: pd.DataFrame      # сводка по кластерам
    risk: risk.RiskResult
    top_zones: pd.DataFrame          # по exposure_rel: где под угрозой больше всего целей
    top_hazard_zones: pd.DataFrame   # по hazard_rel: где опаснее всего лететь
    top_objects: pd.DataFrame
    report: dict                # метрики кластеризации, проверка по облакам, оболочки риска


def run(
    objects: pd.DataFrame,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = config.HDBSCAN_MIN_SAMPLES,
    data_date: Optional[str] = None,
) -> AnalysisResult:
    clustered = objects.reset_index(drop=True).copy()
    labels, cluster_info = clustering.cluster(clustered, min_cluster_size, min_samples)
    clustered["cluster"] = labels

    log.info("Проверка по известным облакам обломков:")
    events = validation.validate_events(clustered)

    risk_result = risk.compute_risk(clustered)
    clustered["hazard_rel"] = risk.object_hazard(clustered, risk_result)

    zones = risk.top_zones(risk_result)
    hazard_zones = risk.top_hazard_zones(risk_result)
    report = {
        "data_date": data_date,
        "note": "Индекс риска относительный (0..1), это не вероятность столкновения.",
        "clustering": cluster_info,
        "events": events,
        "risk": {
            "max_eccentricity": config.RISK_MAX_ECCENTRICITY,
            "grid": {"alt_km": [config.ALT_MIN, config.ALT_MAX, config.ALT_STEP],
                     "inc_deg": [config.INC_MIN, config.INC_MAX, config.INC_STEP]},
            "bands": risk.find_risk_bands(risk_result.profile),
        },
    }
    for title, table in (("Больше всего целей под угрозой", zones),
                         ("Опаснее всего лететь", hazard_zones)):
        if not table.empty:
            best = table.iloc[0]
            log.info("%s: высота %s км, наклонение %s°, целей %d, hazard %.2f, exposure %.2f",
                     title, best["alt_range_km"], best["inc_range_deg"], best["n_targets"],
                     best["hazard_rel"], best["exposure_rel"])
    return AnalysisResult(
        clustered=clustered,
        clusters=clustering.describe_clusters(clustered),
        risk=risk_result,
        top_zones=zones,
        top_hazard_zones=hazard_zones,
        top_objects=risk.top_objects(clustered),
        report=report,
    )


def save(result: AnalysisResult, processed_dir: Optional[Path] = None) -> Dict[str, Path]:
    processed_dir = Path(processed_dir or config.PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        config.CLUSTERED_FILE: result.clustered,
        config.CLUSTERS_FILE: result.clusters,
        config.RISK_GRID_FILE: result.risk.grid,
        config.RISK_PROFILE_FILE: result.risk.profile,
        config.TOP_ZONES_FILE: result.top_zones,
        config.TOP_HAZARD_ZONES_FILE: result.top_hazard_zones,
        config.TOP_OBJECTS_FILE: result.top_objects,
    }
    paths = {}
    for name, df in tables.items():
        paths[name] = processed_dir / name
        df.to_csv(paths[name], index=False)
    paths[config.VALIDATION_FILE] = processed_dir / config.VALIDATION_FILE
    paths[config.VALIDATION_FILE].write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Сохранено в %s: %s", processed_dir, ", ".join(paths))
    return paths
