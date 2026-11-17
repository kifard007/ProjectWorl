"""Свёрточная нейронная сеть для классификации снимков EuroSAT."""

from __future__ import annotations


def build_augmentation(seed: int = 42):
    """Слои аугментации: случайные отражения, повороты и масштаб.

    У спутникового снимка нет «верха» и «низа»: лес, повёрнутый на 90°,
    остаётся лесом. Поэтому отражения и повороты не меняют класс, но
    каждую эпоху показывают сети немного другие картинки. Это снижает
    переобучение. Слои работают только при обучении, а при проверке и
    предсказании пропускают изображение без изменений.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(0.25, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomZoom(0.1, fill_mode="reflect", seed=seed),
        ],
        name="augmentation",
    )


def build_cnn_model(
    input_shape: tuple[int, int, int] = (64, 64, 3),
    n_classes: int = 10,
    learning_rate: float = 1e-3,
    seed: int = 42,
):
    """Создать CNN из трёх свёрточных блоков и полносвязного классификатора.

    TensorFlow импортируется внутри функции, как в лабораторной №4: модуль
    с данными можно использовать и без установленной ML-библиотеки.
    """
    import tensorflow as tf

    layers = tf.keras.layers

    def conv_block(filters: int, name: str):
        # Две свёртки подряд видят область 5×5, но с меньшим числом
        # параметров, чем одна свёртка 5×5. MaxPooling уменьшает карту
        # признаков вдвое (64→32→16→8) и оставляет самые сильные отклики.
        return [
            layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv1"),
            layers.Conv2D(filters, 3, padding="same", activation="relu", name=f"{name}_conv2"),
            layers.MaxPooling2D(2, name=f"{name}_pool"),
        ]

    model = tf.keras.Sequential(
        [
            layers.Input(shape=input_shape),
            build_augmentation(seed),
            # Нормализация пикселей из 0…255 в 0…1 прямо внутри модели:
            # сохранённой модели можно подавать «сырые» картинки.
            layers.Rescaling(1.0 / 255, name="rescaling"),
            *conv_block(32, "block1"),
            *conv_block(64, "block2"),
            *conv_block(128, "block3"),
            layers.Flatten(name="flatten"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(128, activation="relu", name="dense"),
            # Softmax превращает выход в вероятности 10 классов.
            layers.Dense(n_classes, activation="softmax", name="output"),
        ],
        name="eurosat_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        # Метки — целые числа 0…9, поэтому sparse-версия кросс-энтропии.
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model