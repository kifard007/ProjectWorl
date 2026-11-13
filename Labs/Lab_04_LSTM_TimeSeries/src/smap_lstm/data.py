"""Чтение, проверка, нормализация и оконная выборка THULab/SMAP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .download import dataset_paths


EXPECTED_ROWS = {"train": 135_183, "test": 427_617}
EXPECTED_FEATURES = 25


@dataclass
class RawData:
    """Содержимое двух TsFile после проверки схемы."""

    train_values: np.ndarray
    test_values: np.ndarray
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_time: np.ndarray
    test_time: np.ndarray
    feature_names: tuple[str, ...]


@dataclass
class PreparedData:
    """Результаты подготовки многомерного ряда для LSTM."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_target_time: np.ndarray
    test_anomaly_mask: np.ndarray
    feature_names: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler


def _read_tsfile(path: Path) -> pd.DataFrame:
    """Прочитать Apache TsFile через официальный Python SDK."""
    try:
        from tsfile import to_dataframe
    except ImportError as error:
        raise ImportError(
            "Для чтения THULab/SMAP установите зависимости: "
            "pip install -r requirements.txt"
        ) from error
    return to_dataframe(str(path))


def _validate_frame(frame: pd.DataFrame, split: str) -> tuple[str, ...]:
    """Проверить размер, схему, метки и временную ось split."""
    feature_names = tuple(
        sorted(
            (column for column in frame.columns if column.startswith("channel_")),
            key=lambda name: int(name.split("_")[1]),
        )
    )
    required = {"time", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{split}: отсутствуют колонки {required - set(frame.columns)}")
    if len(feature_names) != EXPECTED_FEATURES:
        raise ValueError(
            f"{split}: ожидалось {EXPECTED_FEATURES} каналов, найдено {len(feature_names)}"
        )
    if len(frame) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{split}: ожидалось {EXPECTED_ROWS[split]} строк, найдено {len(frame)}"
        )

    labels = frame["label"].to_numpy(dtype=np.int8)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{split}: label должен содержать только 0/1")
    if split == "train" and labels.any():
        raise ValueError("Train должен содержать только нормальные наблюдения")

    time_values = frame["time"].to_numpy(dtype=np.int64)
    if not np.all(np.diff(time_values) == 1000):
        raise ValueError(f"{split}: временная ось должна иметь шаг 1000 мс")
    return feature_names


def load_dataset(data_dir: str | Path = "data/raw") -> RawData:
    """Прочитать train/test и отделить признаки, время и разметку."""
    train_path, test_path = dataset_paths(data_dir)
    train_frame = _read_tsfile(train_path)
    test_frame = _read_tsfile(test_path)
    train_features = _validate_frame(train_frame, "train")
    test_features = _validate_frame(test_frame, "test")
    if train_features != test_features:
        raise ValueError("Список каналов в train и test различается")

    return RawData(
        train_values=train_frame.loc[:, train_features].to_numpy(dtype=np.float32),
        test_values=test_frame.loc[:, test_features].to_numpy(dtype=np.float32),
        train_labels=train_frame["label"].to_numpy(dtype=np.int8),
        test_labels=test_frame["label"].to_numpy(dtype=np.int8),
        train_time=train_frame["time"].to_numpy(dtype=np.int64),
        test_time=test_frame["time"].to_numpy(dtype=np.int64),
        feature_names=train_features,
    )


def make_sequences(values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Сформировать пары «W прошлых векторов -> следующий вектор» без копии.

    ``sliding_window_view`` создаёт представление над исходным массивом. Для
    SMAP это критично: физическое копирование всех окон потребовало бы гигабайты
    памяти. Перед обучением TensorFlow будет извлекать только текущие батчи.
    """
    if values.ndim != 2:
        raise ValueError("values должен иметь форму (time, features)")
    if window_size <= 0:
        raise ValueError("window_size должен быть положительным")
    if len(values) <= window_size:
        raise ValueError("Длина ряда должна быть больше размера окна")

    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=window_size, axis=0
    )[:-1]
    X = np.moveaxis(windows, -1, 1)
    # Прогнозируем все 25 каналов на следующем временном шаге.
    y = values[window_size:]
    return X, y


def prepare_dataset(
    data_dir: str | Path = "data/raw",
    window_size: int = 100,
    validation_fraction: float = 0.2,
    raw_data: RawData | None = None,
) -> PreparedData:
    """Выполнить всю предобработку без утечки данных из будущего."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction должна находиться между 0 и 1")
    # raw_data позволяет CLI переиспользовать уже прочитанный набор после
    # печати отчёта и не декодировать оба TsFile второй раз.
    raw = raw_data if raw_data is not None else load_dataset(data_dir)

    train_values = np.where(np.isfinite(raw.train_values), raw.train_values, np.nan)
    test_values = np.where(np.isfinite(raw.test_values), raw.test_values, np.nan)
    split_index = int(len(train_values) * (1 - validation_fraction))
    if split_index <= window_size or len(train_values) - split_index <= window_size:
        raise ValueError("После разбиения части ряда слишком коротки для окна")
    train_part = train_values[:split_index]
    val_part = train_values[split_index:]

    # Imputer и scaler оцениваются только по обучающему началу ряда.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(imputer.fit_transform(train_part)).astype(np.float32)
    val_scaled = scaler.transform(imputer.transform(val_part)).astype(np.float32)
    test_scaled = scaler.transform(imputer.transform(test_values)).astype(np.float32)

    X_train, y_train = make_sequences(train_scaled, window_size)
    X_val, y_val = make_sequences(val_scaled, window_size)
    X_test, y_test = make_sequences(test_scaled, window_size)
    target_indices = np.arange(window_size, len(test_values))

    return PreparedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        test_target_time=raw.test_time[target_indices],
        test_anomaly_mask=raw.test_labels[target_indices].astype(bool),
        feature_names=raw.feature_names,
        imputer=imputer,
        scaler=scaler,
    )


def describe_dataset(data_dir: str | Path = "data/raw") -> dict:
    """Вернуть отчёт о форме, типах, пропусках и разметке."""
    return describe_raw_data(load_dataset(data_dir))


def describe_raw_data(raw: RawData) -> dict:
    """Составить отчёт для уже прочитанного набора данных."""
    return {
        "train_shape": raw.train_values.shape,
        "test_shape": raw.test_values.shape,
        "dtype": str(raw.train_values.dtype),
        "features": len(raw.feature_names),
        "feature_names": list(raw.feature_names),
        "train_missing_or_infinite": int((~np.isfinite(raw.train_values)).sum()),
        "test_missing_or_infinite": int((~np.isfinite(raw.test_values)).sum()),
        "train_anomalies": int(raw.train_labels.sum()),
        "test_anomalies": int(raw.test_labels.sum()),
        "time_step_ms": int(raw.train_time[1] - raw.train_time[0]),
    }
