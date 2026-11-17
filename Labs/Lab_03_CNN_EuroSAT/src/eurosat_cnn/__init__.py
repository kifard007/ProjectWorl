"""Классификация спутниковых снимков EuroSAT свёрточной нейронной сетью."""

from .data import CLASS_NAMES, CLASS_NAMES_RU, SplitData, load_eurosat, split_dataset

__all__ = ["CLASS_NAMES", "CLASS_NAMES_RU", "SplitData", "load_eurosat", "split_dataset"]