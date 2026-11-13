"""CLI для пункта 1: получение NASA SMAP."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smap_lstm.download import download_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать датасет THULab/SMAP")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true", help="скачать файлы заново")
    args = parser.parse_args()
    download_dataset(args.data_dir, force=args.force)


if __name__ == "__main__":
    main()

