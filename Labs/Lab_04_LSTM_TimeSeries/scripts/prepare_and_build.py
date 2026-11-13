"""CLI-демонстрация пунктов 2–4: данные, выборки и LSTM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from smap_lstm.data import describe_raw_data, load_dataset, prepare_dataset  # noqa: E402
from smap_lstm.model import build_lstm_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Подготовить SMAP и создать LSTM")
    parser.add_argument("--window", type=int, default=100, help="длина входного окна")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    args = parser.parse_args()

    print("Исходные данные THULab/SMAP:")
    raw_data = load_dataset(args.data_dir)
    print(json.dumps(describe_raw_data(raw_data), indent=2, ensure_ascii=False))
    prepared = prepare_dataset(
        args.data_dir, args.window, args.validation_fraction, raw_data=raw_data
    )
    print("\nФормы подготовленных выборок:")
    for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
        print(f"  {name:8s}: {getattr(prepared, name).shape}")

    model = build_lstm_model(args.window, prepared.X_train.shape[-1])
    print("\nАрхитектура нейронной сети:")
    model.summary()


if __name__ == "__main__":
    main()

