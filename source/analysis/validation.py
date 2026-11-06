"""Проверка кластеров на известных облаках обломков.

Если кластеризация находит настоящие скопления, обломки одного разрушения
(Fengyun-1C, Iridium-33, Cosmos-2251, Cosmos-1408) должны собираться
в одном кластере. Метрика - доля обломков события в самом частом кластере.
"""
from __future__ import annotations

import logging
from typing import List

import pandas as pd

from source import config
from source.analysis.clustering import NOISE

log = logging.getLogger(__name__)


def validate_events(clustered: pd.DataFrame) -> List[dict]:
    """Для каждого облака: куда попали его обломки (DEB) и насколько кластер «чистый»."""
    results = []
    for category, name in config.CLOUD_NAMES.items():
        g = clustered[(clustered["category"] == category) & (clustered["object_type"] == "DEB")]
        row = {"event": name, "category": category, "n_debris": len(g)}
        if len(g) < config.MIN_EVENT_OBJECTS:
            row["status"] = "skipped"
            log.info("  %s: %d обломков на низких орбитах - мало для проверки", name, len(g))
            results.append(row)
            continue

        in_clusters = g[g["cluster"] != NOISE]
        counts = in_clusters["cluster"].value_counts()
        row["status"] = "ok"
        row["noise_share"] = round(1 - len(in_clusters) / len(g), 4)
        if counts.empty:
            row.update(top_cluster=None, top_cluster_share=0.0, n_clusters_used=0, purity=None)
        else:
            top = int(counts.index[0])
            cluster_size = int((clustered["cluster"] == top).sum())
            row.update(
                top_cluster=top,
                # доля всех обломков события в самом частом кластере
                top_cluster_share=round(counts.iloc[0] / len(g), 4),
                n_clusters_used=int(len(counts)),
                # какую долю этого кластера составляют обломки события
                purity=round(counts.iloc[0] / cluster_size, 4),
            )
        log.info("  %s: %.0f%% обломков в кластере %s (чистота %s), шум %.0f%%, обломков %d",
                 name, 100 * row["top_cluster_share"], row["top_cluster"],
                 "-" if row["purity"] is None else f"{row['purity']:.0%}",
                 100 * row["noise_share"], len(g))
        results.append(row)
    return results
