"""Загрузка открытых данных CelesTrak: каталог SATCAT и GP/OMM JSON по группам.

Файлы пишутся в data/raw, сведения о загрузке - в data/raw/meta.json.
Если файл уже скачан, он берётся из кэша (и заново проверяется);
force=True скачивает всё заново.
"""
from __future__ import annotations

import io
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd
import requests

from source import config

log = logging.getLogger(__name__)

# Коды ответа, при которых имеет смысл повторить запрос
RETRY_STATUS = {429, 500, 502, 503, 504}

Fetcher = Callable[[str], bytes]


class LoaderError(RuntimeError):
    """Не удалось скачать или проверить файл."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch(
    url: str,
    *,
    timeout: float = config.HTTP_TIMEOUT_S,
    retries: int = config.HTTP_RETRIES,
    backoff: float = config.HTTP_BACKOFF_S,
) -> bytes:
    """Скачивает url с таймаутом и повторами при сетевых ошибках и 429/5xx."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, timeout=timeout, headers={"User-Agent": config.USER_AGENT}
            )
            if resp.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.content
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code not in RETRY_STATUS:
                raise LoaderError(f"{url}: {exc}") from exc
            last_exc = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
        if attempt < retries:
            delay = backoff * 2 ** (attempt - 1)
            log.warning("%s: %s, повтор через %.0f с (%d/%d)",
                        url, last_exc, delay, attempt, retries)
            time.sleep(delay)
    raise LoaderError(f"{url}: не удалось скачать за {retries} попыток: {last_exc}")


def check_satcat(content: bytes) -> int:
    """Проверяет CSV каталога SATCAT и возвращает число строк."""
    try:
        df = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise LoaderError(f"SATCAT: не похоже на CSV: {exc}") from exc
    missing = [c for c in config.SATCAT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise LoaderError(f"SATCAT: нет колонок {missing}. Возможно, CelesTrak изменил формат.")
    if df.empty:
        raise LoaderError("SATCAT: файл без строк")
    return len(df)


def check_gp(content: bytes, group: str) -> int:
    """Проверяет GP/OMM JSON одной группы и возвращает число объектов."""
    try:
        records = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # На неизвестную группу CelesTrak отвечает текстом, а не JSON
        snippet = content[:80].decode("utf-8", "replace").strip()
        raise LoaderError(f"GP {group}: ответ не JSON: {snippet!r}") from exc
    if not isinstance(records, list):
        raise LoaderError(f"GP {group}: ожидался список объектов")
    if not records:
        log.warning("GP %s: группа пустая", group)
        return 0
    missing = [f for f in config.GP_REQUIRED_FIELDS if f not in records[0]]
    if missing:
        raise LoaderError(f"GP {group}: нет полей {missing}. Возможно, CelesTrak изменил формат.")
    return len(records)


def _write_atomic(path: Path, content: bytes) -> None:
    # Сначала во временный файл: оборванная загрузка не испортит кэш
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def read_meta(raw_dir: Optional[Path] = None) -> dict:
    path = Path(raw_dir or config.RAW_DIR) / config.META_FILE
    if not path.exists():
        return {"source": "CelesTrak", "files": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_file(
    name: str,
    url: str,
    check: Callable[[bytes], int],
    raw_dir: Path,
    force: bool,
    fetcher: Fetcher,
    old_meta: dict,
) -> dict:
    """Скачивает один файл (или берёт из кэша), проверяет и возвращает запись для meta.json."""
    path = raw_dir / name
    if path.exists() and not force:
        content = path.read_bytes()
        rows = check(content)
        prev = old_meta.get("files", {}).get(name, {})
        downloaded_at = prev.get("downloaded_at") or datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
        log.info("%s: из кэша (%d строк, скачан %s)", name, rows, downloaded_at)
        from_cache = True
    else:
        log.info("%s: скачиваю %s", name, url)
        content = fetcher(url)
        rows = check(content)
        _write_atomic(path, content)
        downloaded_at = _now_iso()
        log.info("%s: %d строк, %.1f МБ", name, rows, len(content) / 1e6)
        from_cache = False
    return {
        "url": url,
        "downloaded_at": downloaded_at,
        "rows": rows,
        "bytes": len(content),
        "from_cache": from_cache,
    }


def load_all(
    raw_dir: Optional[Path] = None,
    *,
    force: bool = False,
    groups: Optional[Iterable[str]] = None,
    include_satcat: bool = True,
    fetcher: Fetcher = fetch,
) -> dict:
    """Скачивает SATCAT и GP-группы в raw_dir и записывает meta.json.

    Ошибка в одном файле не останавливает остальные: успешные файлы попадают
    в meta.json, а в конце выбрасывается LoaderError со списком неудачных.
    Возвращает содержимое meta.json.
    """
    raw_dir = Path(raw_dir or config.RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    groups = list(config.GP_GROUPS if groups is None else groups)

    jobs: list = []
    if include_satcat:
        jobs.append((config.SATCAT_FILE, config.SATCAT_URL, check_satcat))
    for group in groups:
        jobs.append((
            config.gp_filename(group),
            config.gp_url(group),
            lambda content, g=group: check_gp(content, g),
        ))

    meta = read_meta(raw_dir)
    failed = []
    for name, url, check in jobs:
        try:
            meta["files"][name] = _load_file(name, url, check, raw_dir, force, fetcher, meta)
        except LoaderError as exc:
            log.error("%s", exc)
            failed.append(name)

    meta["source"] = "CelesTrak"
    meta["updated_at"] = _now_iso()
    (raw_dir / config.META_FILE).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed:
        raise LoaderError(f"Не удалось загрузить: {', '.join(failed)}")
    return meta


def read_satcat(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Читает скачанный SATCAT. Пустые поля (например, DECAY_DATE) становятся NaN."""
    return pd.read_csv(Path(raw_dir or config.RAW_DIR) / config.SATCAT_FILE)


def read_gp(group: str, raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """Читает скачанную GP-группу в DataFrame (одна строка - один объект)."""
    path = Path(raw_dir or config.RAW_DIR) / config.gp_filename(group)
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
