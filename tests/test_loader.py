"""Тесты loader без сети: вместо скачивания подставляется fake_fetcher."""
import json

import pytest

from source import config
from source.loader import celestrak
from source.loader.celestrak import LoaderError, check_gp, check_satcat, load_all

SATCAT_CSV = (
    ",".join(config.SATCAT_REQUIRED_COLUMNS) + "\n"
    + "ISS (ZARYA),1998-067A,25544,PAY,+,,92.9,51.6,420,415,EA\n"
    + "FENGYUN 1C DEB,1999-025A,29000,DEB,,,101.0,98.8,870,840,EA\n"
).encode()

GP_RECORD = {
    "OBJECT_NAME": "ISS (ZARYA)", "OBJECT_ID": "1998-067A", "NORAD_CAT_ID": 25544,
    "EPOCH": "2026-11-05T12:00:00", "MEAN_MOTION": 15.5, "ECCENTRICITY": 0.0005,
    "INCLINATION": 51.6, "RA_OF_ASC_NODE": 120.0,
}


class FakeFetcher:
    """Отдаёт заготовленные ответы и считает вызовы."""

    def __init__(self, gp_answer=None):
        self.calls = []
        self.gp_answer = json.dumps([GP_RECORD]).encode() if gp_answer is None else gp_answer

    def __call__(self, url):
        self.calls.append(url)
        return SATCAT_CSV if url == config.SATCAT_URL else self.gp_answer


def test_download_then_cache(tmp_path):
    fetcher = FakeFetcher()
    meta = load_all(tmp_path, groups=["active"], fetcher=fetcher)

    assert len(fetcher.calls) == 2
    assert (tmp_path / "satcat.csv").read_bytes() == SATCAT_CSV
    assert meta["files"]["satcat.csv"]["rows"] == 2
    assert meta["files"]["gp_active.json"]["rows"] == 1
    assert not meta["files"]["satcat.csv"]["from_cache"]

    meta2 = load_all(tmp_path, groups=["active"], fetcher=fetcher)
    assert len(fetcher.calls) == 2, "повторный запуск не должен ходить в сеть"
    assert meta2["files"]["satcat.csv"]["from_cache"]
    # Дата скачивания сохраняется из прошлого meta.json
    assert (meta2["files"]["satcat.csv"]["downloaded_at"]
            == meta["files"]["satcat.csv"]["downloaded_at"])


def test_force_redownloads(tmp_path):
    fetcher = FakeFetcher()
    load_all(tmp_path, groups=["active"], fetcher=fetcher)
    load_all(tmp_path, groups=["active"], fetcher=fetcher, force=True)
    assert len(fetcher.calls) == 4


def test_meta_json_written(tmp_path):
    load_all(tmp_path, groups=["active"], fetcher=FakeFetcher())
    meta = json.loads((tmp_path / config.META_FILE).read_text(encoding="utf-8"))
    assert meta["source"] == "CelesTrak"
    assert meta["files"]["gp_active.json"]["url"] == config.gp_url("active")
    assert "updated_at" in meta


def test_bad_group_does_not_stop_others(tmp_path):
    fetcher = FakeFetcher(gp_answer=b'Invalid query: "GROUP=xxx" not found')
    with pytest.raises(LoaderError, match="gp_xxx.json"):
        load_all(tmp_path, groups=["xxx"], fetcher=fetcher)
    # SATCAT при этом скачан и записан в meta.json, а битый файл - нет
    assert (tmp_path / "satcat.csv").exists()
    assert not (tmp_path / "gp_xxx.json").exists()
    meta = json.loads((tmp_path / config.META_FILE).read_text(encoding="utf-8"))
    assert list(meta["files"]) == ["satcat.csv"]


def test_check_satcat_rejects_missing_columns():
    with pytest.raises(LoaderError, match="нет колонок"):
        check_satcat(b"OBJECT_NAME,NORAD_CAT_ID\nISS,25544\n")


def test_check_gp():
    assert check_gp(json.dumps([GP_RECORD, GP_RECORD]).encode(), "active") == 2
    assert check_gp(b"[]", "empty") == 0
    with pytest.raises(LoaderError, match="нет полей"):
        check_gp(json.dumps([{"OBJECT_NAME": "X"}]).encode(), "active")


def test_fetch_retries_then_fails(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        raise celestrak.requests.ConnectionError("нет сети")

    monkeypatch.setattr(celestrak.requests, "get", fake_get)
    monkeypatch.setattr(celestrak.time, "sleep", lambda s: None)
    with pytest.raises(LoaderError, match="3 попыток"):
        celestrak.fetch("https://example.invalid", retries=3)
    assert len(calls) == 3
