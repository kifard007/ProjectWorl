"""Архитектура LSTM для прогноза следующего 25-мерного наблюдения."""

from __future__ import annotations


def build_lstm_model(window_size: int, n_features: int, learning_rate: float = 1e-3):
    """Создать двухслойную LSTM-регрессию.

    TensorFlow импортируется внутри функции: анализ и подготовка данных могут
    работать даже в окружении, где тяжёлая ML-зависимость ещё не установлена.
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            # Первый слой возвращает всю последовательность скрытых состояний,
            # потому что её должен обработать следующий рекуррентный слой.
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            # Второй LSTM сворачивает окно в один вектор признаков.
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation="relu"),
            # Линейный выход прогнозирует все каналы на следующем шаге.
            tf.keras.layers.Dense(n_features),
        ],
        name="smap_lstm_forecaster",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model

