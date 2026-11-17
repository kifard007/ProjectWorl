"""Запуск всей лабораторной: данные, CNN, обучение, тест, графики."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from eurosat_cnn.data import describe_dataset, load_eurosat  # noqa: E402
from eurosat_cnn.experiment import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучить CNN на EuroSAT RGB")
    parser.add_argument("--epochs", type=int, default=30, help="максимум эпох")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument(
        "--quick", action="store_true",
        help="быстрая проверка: 3000 снимков, 2 эпохи, результаты в results_quick/",
    )
    args = parser.parse_args()

    images, labels = load_eurosat(args.data_dir)
    print(json.dumps(describe_dataset(images, labels), indent=2, ensure_ascii=False))

    if args.quick:
        rng = np.random.default_rng(42)
        subset = rng.choice(len(images), 3000, replace=False)
        images, labels = images[subset], labels[subset]
        args.epochs = 2
        args.results_dir = PROJECT_ROOT / "results_quick"

    result = run_experiment(
        images, labels, args.results_dir, epochs=args.epochs, batch_size=args.batch_size
    )
    print("\nИтог на тестовой выборке:")
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    print(f"\nГрафики и метрики сохранены в {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()