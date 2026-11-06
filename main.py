"""Точка входа.

python main.py        - запускает приложение (GUI пока в разработке)
python main.py --cli  - прогоняет конвейер без окна
"""
from __future__ import annotations

import argparse
import logging
import sys

from source import config


def run_cli(force: bool) -> int:
    from source.loader.celestrak import LoaderError, load_all

    log = logging.getLogger("main")
    try:
        load_all(force=force)
    except LoaderError as exc:
        log.error("%s", exc)
        return 1
    # Следующие шаги (processing, analysis, visualization) появятся здесь
    log.info("Конвейер: пока реализована только загрузка данных")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Карта риска космического мусора")
    parser.add_argument("--cli", action="store_true", help="запустить конвейер без окна")
    parser.add_argument("--force", action="store_true", help="скачать данные заново")
    args = parser.parse_args(argv)

    config.setup_logging()
    if args.cli:
        return run_cli(args.force)
    logging.getLogger("main").error("GUI ещё не готов, запустите: python main.py --cli")
    return 2


if __name__ == "__main__":
    sys.exit(main())
