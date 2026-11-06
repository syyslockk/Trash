"""Кластеризация объектов HDBSCAN по высоте и наклонению.

Почему HDBSCAN, а не k-means: число групп заранее неизвестно, плотность
у групп разная, а часть объектов ни в какую группу не входит (шум, метка -1).
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from source import config

log = logging.getLogger(__name__)

NOISE = -1


def _make_model(min_cluster_size: int, min_samples: int):
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples), "hdbscan"
    except ImportError:
        # На Apple Silicon пакет hdbscan иногда не собирается
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=min_cluster_size,
                       min_samples=min_samples), "sklearn"


def cluster(
    objects: pd.DataFrame,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = config.HDBSCAN_MIN_SAMPLES,
    features: Sequence[str] = tuple(config.CLUSTER_FEATURES),
) -> Tuple[np.ndarray, dict]:
    """Возвращает метки кластеров (-1 - шум) и сводку с метриками качества."""
    X = StandardScaler().fit_transform(objects[list(features)].to_numpy(dtype=float))
    model, backend = _make_model(min_cluster_size, min_samples)
    labels = model.fit_predict(X)

    clustered = labels != NOISE
    n_clusters = len(set(labels[clustered]))
    n_noise = int((~clustered).sum())
    info = {
        "backend": backend,
        "features": list(features),
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_objects": int(len(labels)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_share": round(n_noise / len(labels), 4) if len(labels) else 0.0,
        "silhouette": None,
        "davies_bouldin": None,
    }
    if n_clusters > 1:
        sample = min(config.SILHOUETTE_SAMPLE, int(clustered.sum()))
        info["silhouette"] = round(float(silhouette_score(
            X[clustered], labels[clustered],
            sample_size=sample, random_state=config.RANDOM_SEED)), 4)
        info["davies_bouldin"] = round(float(davies_bouldin_score(
            X[clustered], labels[clustered])), 4)

    log.info("HDBSCAN (%s): кластеров %d, шум %d (%.1f%%)",
             backend, n_clusters, n_noise, 100 * info["noise_share"])
    if info["silhouette"] is not None:
        log.info("  Silhouette %.3f (ближе к 1 - лучше), Davies-Bouldin %.3f (ближе к 0 - лучше)",
                 info["silhouette"], info["davies_bouldin"])
    return labels, info


def describe_clusters(clustered: pd.DataFrame, hazard_col: Optional[str] = "hazard_rel") -> pd.DataFrame:
    """Сводка по кластерам: размер, преобладающая категория, медианы высоты и наклонения."""
    rows = []
    for cid, g in clustered[clustered["cluster"] != NOISE].groupby("cluster"):
        dom = g["category"].value_counts()
        row = {
            "cluster": int(cid),
            "n_objects": len(g),
            "n_debris": int((g["object_type"] == "DEB").sum()),
            "dominant_category": dom.index[0],
            "dominant_share": round(dom.iloc[0] / len(g), 3),
            "median_alt_km": round(g["mean_alt_km"].median(), 1),
            "median_inc_deg": round(g["inclination_deg"].median(), 2),
        }
        if hazard_col and hazard_col in g:
            row["mean_hazard_rel"] = round(g[hazard_col].mean(), 4)
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["cluster", "n_objects"])
    return pd.DataFrame(rows).sort_values("n_objects", ascending=False).reset_index(drop=True)
