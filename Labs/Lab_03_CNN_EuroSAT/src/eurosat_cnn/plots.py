"""Графики и таблицы для анализа результатов."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # сохранение в файл без открытия окон
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from .data import CLASS_NAMES, CLASS_NAMES_RU  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_class_examples(images, labels, path: Path, per_class: int = 5, seed: int = 42) -> Path:
    """Сетка примеров: по ``per_class`` снимков каждого класса."""
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(CLASS_NAMES), per_class, figsize=(per_class * 1.6, 17))
    for row, name in enumerate(CLASS_NAMES_RU):
        indices = rng.choice(np.flatnonzero(labels == row), per_class, replace=False)
        for col, index in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[index])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle("Примеры снимков EuroSAT по классам")
    return _save(fig, path)


def plot_class_distribution(labels, path: Path) -> Path:
    """Столбчатая диаграмма числа изображений в каждом классе."""
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(CLASS_NAMES_RU, counts, color="#4C78A8")
    ax.set_ylabel("Количество изображений")
    ax.set_title("Распределение изображений по классам")
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return _save(fig, path)


def plot_augmentation(image, augmentation, path: Path, n: int = 8) -> Path:
    """Один снимок и несколько его случайно аугментированных вариантов."""
    batch = np.repeat(image[None, ...], n, axis=0)
    augmented = augmentation(batch, training=True).numpy()
    fig, axes = plt.subplots(1, n + 1, figsize=((n + 1) * 1.7, 2.2))
    axes[0].imshow(image)
    axes[0].set_title("Оригинал", fontsize=9)
    for i in range(n):
        axes[i + 1].imshow(np.clip(augmented[i], 0, 255).astype(np.uint8))
        axes[i + 1].set_title(f"Вариант {i + 1}", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Аугментация: отражения, повороты, масштаб")
    return _save(fig, path)


def plot_history(history: dict, path: Path) -> Path:
    """Графики точности и ошибки на обучении и валидации по эпохам."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_acc.plot(epochs, history["accuracy"], label="Обучение")
    ax_acc.plot(epochs, history["val_accuracy"], label="Валидация")
    ax_acc.set_title("Точность (accuracy)")
    ax_loss.plot(epochs, history["loss"], label="Обучение")
    ax_loss.plot(epochs, history["val_loss"], label="Валидация")
    ax_loss.set_title("Функция потерь")
    for ax in (ax_acc, ax_loss):
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.3)
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(y_true, y_pred, path: Path) -> Path:
    """Матрица ошибок в долях: строка — истинный класс, столбец — предсказанный."""
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ticks = np.arange(len(CLASS_NAMES))
    ax.set_xticks(ticks, CLASS_NAMES_RU, rotation=45, ha="right")
    ax.set_yticks(ticks, CLASS_NAMES_RU)
    for i in ticks:
        for j in ticks:
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.5 else "black")
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок на тестовой выборке")
    return _save(fig, path)


def plot_misclassified(images, y_true, y_pred, probabilities, path: Path, n: int = 16) -> Path:
    """Снимки, на которых модель ошиблась, с самой уверенной ошибкой первыми."""
    wrong = np.flatnonzero(y_true != y_pred)
    wrong = wrong[np.argsort(-probabilities[wrong].max(axis=1))][:n]
    cols = 4
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, index in zip(axes.ravel(), wrong):
        ax.imshow(images[index])
        ax.set_title(
            f"Истина: {CLASS_NAMES_RU[y_true[index]]}\n"
            f"Модель: {CLASS_NAMES_RU[y_pred[index]]} ({probabilities[index].max():.0%})",
            fontsize=8,
        )
    fig.suptitle("Примеры ошибок модели")
    return _save(fig, path)


def classification_table(y_true, y_pred) -> pd.DataFrame:
    """Precision, recall и F1 по каждому классу."""
    report = classification_report(
        y_true, y_pred, labels=range(len(CLASS_NAMES)), target_names=CLASS_NAMES_RU,
        output_dict=True, zero_division=0,
    )
    table = pd.DataFrame(report).T
    table["support"] = table["support"].astype(int)
    return table.round(4)