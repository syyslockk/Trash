"""python -m source.processing [-v]"""
from __future__ import annotations

import argparse
import logging
import sys

from source import config
from source.processing.clean import ProcessingError, process, save


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m source.processing",
        description="Объединяет SATCAT и GP из data/raw в data/processed/objects.csv.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="подробный лог")
    args = parser.parse_args(argv)

    config.setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        save(process())
    except ProcessingError as exc:
        logging.getLogger("source.processing").error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
