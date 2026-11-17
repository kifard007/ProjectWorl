"""Скачать EuroSAT RGB с Hugging Face (timm/eurosat-rgb) в папку data/."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from eurosat_cnn.data import describe_dataset, load_eurosat  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать и проверить EuroSAT RGB")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()

    print("Загрузка EuroSAT RGB (~90 МБ, при первом запуске)...")
    images, labels = load_eurosat(args.data_dir)
    print(json.dumps(describe_dataset(images, labels), indent=2, ensure_ascii=False))
    print(f"Датасет готов: {args.data_dir.resolve()}")


if __name__ == "__main__":
    main()