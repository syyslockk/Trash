"""python -m source.analysis [--min-cluster-size N] [--min-samples N] [-v]"""
from __future__ import annotations

import argparse
import logging
import sys

from source import config
from source.analysis.pipeline import run, save
from source.loader.celestrak import read_meta


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m source.analysis",
        description="Кластеризация, проверка по облакам обломков и сетка риска.",
    )
    parser.add_argument("--min-cluster-size", type=int, default=config.HDBSCAN_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-samples", type=int, default=config.HDBSCAN_MIN_SAMPLES)
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args(argv)

    config.setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    log = logging.getLogger("source.analysis")
    path = config.PROCESSED_DIR / config.OBJECTS_FILE
    if not path.exists():
        log.error("Нет %s. Сначала запустите: python -m source.processing", path)
        return 1

    from source.processing.clean import read_objects
    data_date = read_meta().get("files", {}).get(config.SATCAT_FILE, {}).get("downloaded_at")
    result = run(read_objects(), args.min_cluster_size, args.min_samples, data_date=data_date)
    save(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
