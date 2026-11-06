"""Тесты processing на маленьком искусственном каталоге."""
import numpy as np
import pandas as pd
import pytest

from source import config
from source.processing.clean import (
    OBJECT_COLUMNS, ProcessingError, build_objects, filter_leo, kepler_heights, process,
)


def satcat_row(norad, name, obj_type, ops="", apogee=420.0, perigee=415.0, inc=51.6,
               decay=None, center="EA", orbit_type="ORB"):
    return {
        "OBJECT_NAME": name, "OBJECT_ID": f"2000-{norad}A", "NORAD_CAT_ID": norad,
        "OBJECT_TYPE": obj_type, "OPS_STATUS_CODE": ops or np.nan, "DECAY_DATE": decay,
        "PERIOD": 92.9, "INCLINATION": inc, "APOGEE": apogee, "PERIGEE": perigee,
        "ORBIT_CENTER": center, "ORBIT_TYPE": orbit_type,
    }


def gp_row(norad, name, mean_motion=15.5, ecc=0.0005, inc=51.6, raan=120.0):
    return {
        "OBJECT_NAME": name, "OBJECT_ID": f"2000-{norad}A", "NORAD_CAT_ID": norad,
        "EPOCH": "2026-11-05T12:00:00", "MEAN_MOTION": mean_motion, "ECCENTRICITY": ecc,
        "INCLINATION": inc, "RA_OF_ASC_NODE": raan,
    }


@pytest.fixture
def objects():
    satcat = pd.DataFrame([
        satcat_row(1, "ISS", "PAY", ops="+"),
        satcat_row(2, "OLD SAT", "PAY", ops="-", apogee=800, perigee=780, inc=98.0),
        satcat_row(3, "SOME R/B", "R/B", apogee=650, perigee=600, inc=97.5),
        satcat_row(4, "FENGYUN 1C DEB", "DEB", apogee=870, perigee=840, inc=98.8),
        satcat_row(5, "OTHER DEB", "DEB", apogee=1200, perigee=1100, inc=74.0),
        satcat_row(6, "DECAYED DEB", "DEB", decay="2020-01-01", orbit_type="IMP"),
        satcat_row(7, "GEO SAT", "PAY", ops="+", apogee=35800, perigee=35780, inc=0.1),
        satcat_row(8, "NO ORBIT", "DEB", apogee=np.nan, perigee=np.nan),
        satcat_row(9, "MOON PROBE", "PAY", ops="+", center="MO"),
        satcat_row(10, "NEW R/B", "R/B"),
    ])
    gp = {
        "active": pd.DataFrame([gp_row(1, "ISS"), gp_row(10, "NEW R/B")]),
        "fengyun-1c-debris": pd.DataFrame([
            gp_row(4, "FENGYUN 1C DEB", mean_motion=14.2, ecc=0.002, inc=98.8, raan=30.0),
        ]),
    }
    return build_objects(satcat, gp).set_index("norad_id")


def test_kepler_heights_iss():
    # ISS: около 15.5 об/сут -> высота примерно 415-420 км
    h = kepler_heights([15.5], [0.0005]).iloc[0]
    assert 400 < h["mean_alt_km"] < 430
    assert h["perigee_km"] < h["mean_alt_km"] < h["apogee_km"]


def test_gp_heights_match_satcat(objects):
    # Высоты по Кеплеру совпадают с APOGEE/PERIGEE из SATCAT с точностью ~20 км
    assert objects.loc[1, "source"] == "gp"
    assert abs(objects.loc[1, "mean_alt_km"] - 417.5) < 20


def test_drops_decayed_offorbit_and_empty(objects):
    for norad in (6, 8, 9):
        assert norad not in objects.index


def test_columns_and_no_duplicates(objects):
    assert list(objects.reset_index().columns) == OBJECT_COLUMNS
    assert objects.index.is_unique


def test_categories(objects):
    cat = objects["category"]
    assert cat[1] == "active_satellite"
    assert cat[2] == "inactive_payload"
    assert cat[3] == "rocket_body"
    assert cat[4] == "fengyun_1c_debris"
    assert cat[5] == "debris"
    assert cat[10] == "rocket_body"


def test_is_active(objects):
    assert objects.loc[1, "is_active"]
    assert not objects.loc[2, "is_active"]
    # Ступень из группы active действующей не становится
    assert not objects.loc[10, "is_active"]


def test_raan_only_from_gp(objects):
    assert objects.loc[4, "raan_deg"] == 30.0
    assert pd.isna(objects.loc[5, "raan_deg"])
    assert objects.loc[5, "source"] == "satcat"


def test_satcat_eccentricity(objects):
    r_apo, r_per = config.R_EARTH + 1200, config.R_EARTH + 1100
    assert objects.loc[5, "eccentricity"] == pytest.approx((r_apo - r_per) / (r_apo + r_per))


def test_filter_leo(objects):
    leo = filter_leo(objects.reset_index())
    assert 7 not in set(leo["norad_id"])
    assert leo["mean_alt_km"].between(config.ALT_MIN, config.ALT_MAX).all()


def test_process_without_raw_data(tmp_path):
    with pytest.raises(ProcessingError, match="source.loader"):
        process(tmp_path)
