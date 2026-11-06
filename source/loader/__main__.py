"""python -m source.loader [--force] [--groups G ...] [--no-satcat] [-v]"""
from __future__ import annotations

import argparse
import logging
import sys

from source import config
from source.loader.celestrak import LoaderError, load_all


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m source.loader",
        description="Скачивает SATCAT и GP-группы CelesTrak в data/raw.",
    )
    parser.add_argument("--force", action="store_true",
                        help="скачать заново, даже если файлы уже есть")
    parser.add_argument("--groups", nargs="+", metavar="G",
                        help=f"GP-группы (по умолчанию: {' '.join(config.GP_GROUPS)})")
    parser.add_argument("--no-satcat", action="store_true", help="не загружать SATCAT")
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args(argv)

    config.setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        meta = load_all(force=args.force, groups=args.groups,
                        include_satcat=not args.no_satcat)
    except LoaderError as exc:
        logging.getLogger("source.loader").error("%s", exc)
        return 1
    total = sum(f["rows"] for f in meta["files"].values())
    logging.getLogger("source.loader").info(
        "Готово: %d файлов, %d строк, папка %s", len(meta["files"]), total, config.RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
