"""Классификация спутниковых снимков EuroSAT свёрточной нейронной сетью."""

from .data import CLASS_NAMES, CLASS_NAMES_RU, SplitData, load_eurosat, split_dataset
from .model import build_augmentation, build_cnn_model

__all__ = [
    "CLASS_NAMES",
    "CLASS_NAMES_RU",
    "SplitData",
    "load_eurosat",
    "split_dataset",
    "build_augmentation",
    "build_cnn_model",
]