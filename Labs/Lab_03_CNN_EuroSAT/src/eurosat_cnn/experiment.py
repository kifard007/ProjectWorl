"""Полный эксперимент: выборки → обучение → тест → метрики и графики."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import plots
from .data import CLASS_NAMES, make_tf_dataset, split_dataset
from .model import build_cnn_model


def run_experiment(
    images: np.ndarray,
    labels: np.ndarray,
    results_dir: str | Path = "results",
    epochs: int = 30,
    batch_size: int = 64,
    patience: int = 5,
    seed: int = 42,
) -> dict:
    """Выполнить пункты 3–7 лабораторной и сохранить результаты в ``results_dir``."""
    import tensorflow as tf

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(seed)

    # 2. Наглядный обзор данных.
    plots.plot_class_examples(images, labels, results_dir / "class_examples.png", seed=seed)
    plots.plot_class_distribution(labels, results_dir / "class_distribution.png")

    # 3. Подготовка выборок: train/test 80/20, затем 20% train на validation.
    split = split_dataset(images, labels, seed=seed)
    sizes = {"train": len(split.X_train), "validation": len(split.X_val), "test": len(split.X_test)}
    print("Размеры выборок:", sizes)
    train_ds = make_tf_dataset(split.X_train, split.y_train, batch_size, shuffle=True, seed=seed)
    val_ds = make_tf_dataset(split.X_val, split.y_val, batch_size)
    test_ds = make_tf_dataset(split.X_test, split.y_test, batch_size)

    # 4. Создание CNN.
    model = build_cnn_model(input_shape=images.shape[1:], n_classes=len(CLASS_NAMES), seed=seed)
    model.summary()
    summary_lines: list[str] = []
    model.summary(print_fn=summary_lines.append)
    (results_dir / "model_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    plots.plot_augmentation(
        split.X_train[0], model.get_layer("augmentation"), results_dir / "augmentation_examples.png"
    )

    # 5. Обучение с контролем по validation.
    callbacks = [
        # Остановиться, если val_loss не улучшается patience эпох подряд,
        # и вернуть веса лучшей эпохи.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
        ),
        # Уменьшить шаг обучения вдвое, когда ошибка «застряла».
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            results_dir / "best_model.keras", monitor="val_loss", save_best_only=True
        ),
    ]
    start = time.perf_counter()
    # shuffle=False: train_ds уже перемешивается внутри tf.data каждую эпоху.
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, shuffle=False
    )
    train_seconds = time.perf_counter() - start

    history_frame = pd.DataFrame(history.history)
    history_frame.index = history_frame.index + 1
    history_frame.to_csv(results_dir / "history.csv", index_label="epoch")
    plots.plot_history(history.history, results_dir / "training_history.png")

    # 6. Проверка на тестовой выборке, которую модель не видела.
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = probabilities.argmax(axis=1)

    # 7. Метрики и анализ ошибок.
    report = plots.classification_table(split.y_test, y_pred)
    report.to_csv(results_dir / "classification_report.csv", index_label="class")
    plots.plot_confusion_matrix(split.y_test, y_pred, results_dir / "confusion_matrix.png")
    plots.plot_misclassified(
        split.X_test, split.y_test, y_pred, probabilities, results_dir / "misclassified.png"
    )

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "macro_f1": float(report.loc["macro avg", "f1-score"]),
        "errors": int((split.y_test != y_pred).sum()),
        "test_images": int(len(split.y_test)),
        "epochs_trained": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "train_minutes": round(train_seconds / 60, 1),
    }
    pd.Series(metrics, name="value").to_csv(results_dir / "test_metrics.csv", index_label="metric")
    return {"metrics": metrics, "report": report, "history": history.history, "split": split,
            "model": model, "y_pred": y_pred, "probabilities": probabilities}