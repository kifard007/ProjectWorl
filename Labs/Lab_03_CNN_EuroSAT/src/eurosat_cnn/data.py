"""Загрузка, проверка и разделение EuroSAT RGB на выборки."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


# Копия EuroSAT RGB на Hugging Face: те же 27 000 снимков и 10 классов,
# что и в TensorFlow Datasets (eurosat/rgb).
DATASET_REPO = "timm/eurosat-rgb"
IMAGE_SHAPE = (64, 64, 3)
EXPECTED_IMAGES = 27_000

# Порядок классов совпадает с разметкой датасета (label 0…9).
CLASS_NAMES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
CLASS_NAMES_RU = (
    "Однолетние культуры",
    "Лес",
    "Травянистая растительность",
    "Шоссе",
    "Промышленная зона",
    "Пастбище",
    "Многолетние культуры",
    "Жилая застройка",
    "Река",
    "Море/озеро",
)


@dataclass
class SplitData:
    """Три непересекающиеся выборки изображений и меток."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def download_eurosat(data_dir: str | Path = "data") -> Path:
    """Скачать parquet-файлы EuroSAT RGB с Hugging Face (~90 МБ).

    Повторный запуск ничего не скачивает заново: уже загруженные файлы
    берутся из ``data_dir``.
    """
    from huggingface_hub import snapshot_download

    target = Path(data_dir) / "eurosat-rgb"
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=target,
    )
    return target


def load_eurosat(data_dir: str | Path = "data") -> tuple[np.ndarray, np.ndarray]:
    """Скачать (при первом запуске) и прочитать EuroSAT RGB целиком.

    В репозитории снимки разложены на три файла (train/validation/test).
    Мы объединяем их обратно в один набор из 27 000 снимков и делим
    по-своему в ``split_dataset``. Изображения возвращаются как uint8
    (0…255): так весь набор занимает ~330 МБ вместо ~1,3 ГБ во float32, а
    приведение к диапазону 0…1 выполняет слой Rescaling внутри модели.
    """
    import pandas as pd
    from PIL import Image

    files = sorted((download_eurosat(data_dir) / "data").glob("*.parquet"))
    if len(files) != 3:
        raise FileNotFoundError(f"Ожидалось 3 parquet-файла, найдено {len(files)}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)

    # Каждый снимок хранится как JPEG; декодируем его в массив 64×64×3.
    images = np.stack(
        [np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in frame["image"]]
    )
    labels = frame["label"].to_numpy()
    if len(images) != EXPECTED_IMAGES:
        raise ValueError(f"Ожидалось {EXPECTED_IMAGES} изображений, получено {len(images)}")
    validate_images(images, labels)
    return images, labels.astype(np.int64)


def validate_images(images: np.ndarray, labels: np.ndarray) -> None:
    """Проверить размер, тип и метки набора."""
    if images.ndim != 4 or images.shape[1:] != IMAGE_SHAPE:
        raise ValueError(f"Ожидались изображения {IMAGE_SHAPE}, получено {images.shape[1:]}")
    if images.dtype != np.uint8:
        raise ValueError(f"Ожидался тип uint8, получен {images.dtype}")
    if len(images) != len(labels):
        raise ValueError("Число изображений и меток различается")
    if labels.min() < 0 or labels.max() >= len(CLASS_NAMES):
        raise ValueError("Метки должны быть в диапазоне 0…9")


def describe_dataset(images: np.ndarray, labels: np.ndarray) -> dict:
    """Короткий отчёт о наборе данных."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    return {
        "Всего изображений": int(len(images)),
        "Размер изображения": tuple(int(v) for v in images.shape[1:]),
        "Тип пикселей": str(images.dtype),
        "Диапазон значений": (int(images.min()), int(images.max())),
        "Количество классов": len(CLASS_NAMES),
        "Изображений по классам": {
            name: int(count) for name, count in zip(CLASS_NAMES, counts)
        },
    }


def split_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> SplitData:
    """Разделить набор на train/test, затем отрезать validation от train.

    Сначала делим весь набор на train и test (80/20). Затем, как в лабораторной №4, 20% обучающей
    части отводим на validation. Итого 64% / 16% / 20%.

    Стратификация сохраняет одинаковую долю каждого класса во всех частях,
    а фиксированный ``seed`` делает разбиение воспроизводимым.
    """
    if not 0 < test_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Доли test и validation должны быть между 0 и 1")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        images, labels, test_size=test_fraction, stratify=labels, random_state=seed
    )
    # validation_fraction — доля от обучающей части, а не от всего набора.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=validation_fraction,
        stratify=y_train_full, random_state=seed,
    )
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test)


def make_tf_dataset(images, labels, batch_size: int = 64, shuffle: bool = False, seed: int = 42):
    """Обернуть массивы в tf.data для подачи модели батчами."""
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(len(images), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)