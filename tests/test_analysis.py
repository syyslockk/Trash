"""Тесты analysis на искусственном наборе: два облака обломков и спутники."""
import numpy as np
import pandas as pd
import pytest

from source import config
from source.analysis import pipeline, risk
from source.analysis.clustering import NOISE, cluster
from source.analysis.validation import validate_events


def make_objects(seed=0):
    rng = np.random.default_rng(seed)

    def group(n, alt, inc, object_type, category, active=False, ecc=0.001):
        return pd.DataFrame({
            "norad_id": 0,
            "object_name": category.upper(),
            "object_type": object_type,
            "category": category,
            "is_active": active,
            "mean_alt_km": rng.normal(alt, 5, n),
            "inclination_deg": rng.normal(inc, 0.3, n),
            "eccentricity": ecc,
        })

    df = pd.concat([
        group(300, 850, 98.8, "DEB", "fengyun_1c_debris"),
        group(200, 790, 74.0, "DEB", "cosmos_2251_debris"),
        group(400, 550, 53.0, "PAY", "active_satellite", active=True),
        group(100, 600, 97.5, "R/B", "rocket_body"),
        group(10, 700, 30.0, "DEB", "iridium_33_debris"),
        group(5, 900, 60.0, "DEB", "debris", ecc=0.3),
    ], ignore_index=True)
    df["norad_id"] = np.arange(1, len(df) + 1)
    return df


@pytest.fixture(scope="module")
def result():
    return pipeline.run(make_objects())


def test_vrel_matrix_properties():
    inc = risk.make_grid().inc_centers
    F = risk.vrel_factor_matrix(inc)
    assert F.shape == (len(inc), len(inc))
    assert np.allclose(F, F.T)
    assert (F >= 0).all() and (F <= 2).all()
    assert (np.diag(F) > 0).all()
    # Встречные орбиты (i и 180-i) сближаются быстрее, чем одинаковые
    assert F[0, -1] > F[0, 0]


def test_risk_indices_in_unit_range(result):
    grid = result.risk.grid
    for col in ("hazard_rel", "exposure_rel"):
        assert grid[col].between(0, 1).all()
        assert grid[col].max() == pytest.approx(1.0)
    assert result.clustered["hazard_rel"].between(0, 1).all()
    assert result.risk.profile["exposure_rel"].between(0, 1).all()


def test_grid_counts(result):
    grid = result.risk.grid
    assert len(grid) == len(risk.make_grid().alt_centers) * len(risk.make_grid().inc_centers)
    # 300 + 200 + 10 обломков; вытянутые орбиты (e = 0.3) в риск не идут
    assert grid["n_debris"].sum() == 510
    assert grid["n_active_satellites"].sum() == 400
    assert grid["n_rocket_bodies"].sum() == 100


def test_hazard_highest_near_debris(result):
    by_cat = result.clustered.groupby("category")["hazard_rel"].mean()
    assert by_cat["fengyun_1c_debris"] > by_cat["active_satellite"]


def test_clusters_recover_clouds(result):
    events = {e["category"]: e for e in result.report["events"]}
    assert events["fengyun_1c_debris"]["top_cluster_share"] > 0.9
    assert events["cosmos_2251_debris"]["top_cluster_share"] > 0.9
    assert events["fengyun_1c_debris"]["top_cluster"] != events["cosmos_2251_debris"]["top_cluster"]
    # 10 обломков меньше порога MIN_EVENT_OBJECTS
    assert events["iridium_33_debris"]["status"] == "skipped"


def test_clustering_metrics(result):
    info = result.report["clustering"]
    assert info["n_clusters"] >= 4
    assert -1 <= info["silhouette"] <= 1
    assert info["davies_bouldin"] >= 0


def test_clustering_is_deterministic():
    df = make_objects()
    labels1, info1 = cluster(df)
    labels2, info2 = cluster(df)
    assert (labels1 == labels2).all()
    assert info1 == info2


def test_top_tables(result):
    zones = result.top_zones
    assert 0 < len(zones) <= config.TOP_N
    assert zones["exposure_rel"].is_monotonic_decreasing
    assert zones["n_targets"].gt(0).all()

    hazard_zones = result.top_hazard_zones
    assert 0 < len(hazard_zones) <= config.TOP_N
    assert hazard_zones["hazard_rel"].is_monotonic_decreasing
    assert hazard_zones["n_targets"].ge(config.HAZARD_ZONE_MIN_TARGETS).all()

    top = result.top_objects
    assert set(top["object_type"]) <= {"PAY", "R/B"}
    assert top["hazard_rel"].is_monotonic_decreasing


def test_risk_bands():
    profile = pd.DataFrame({
        "alt_center_km": [170.0, 190.0, 210.0, 230.0, 250.0],
        "exposure_rel": [0.1, 0.5, 1.0, 0.1, 0.3],
    })
    bands = risk.find_risk_bands(profile)
    assert [(b["lo_km"], b["hi_km"]) for b in bands] == [(180.0, 220.0), (240.0, 260.0)]


def test_validation_counts_only_debris():
    df = make_objects()
    df["cluster"] = NOISE
    events = {e["category"]: e for e in validate_events(df)}
    assert events["fengyun_1c_debris"]["noise_share"] == 1.0
    assert events["fengyun_1c_debris"]["top_cluster"] is None


def test_save_writes_all_files(result, tmp_path):
    paths = pipeline.save(result, tmp_path)
    for name in (config.CLUSTERED_FILE, config.RISK_GRID_FILE, config.TOP_ZONES_FILE,
                 config.TOP_HAZARD_ZONES_FILE,
                 config.TOP_OBJECTS_FILE, config.VALIDATION_FILE):
        assert paths[name].exists()
