"""Definição funcional das arquiteturas neurais baseline:

- MLP (Perceptron Multicamadas com vetorização de tensores RGB)
- CNN (Rede Neural Convolucional com GlobalAveragePooling2D para invariância dimensional)
"""

from typing import Any


def build_mlp(
    input_shape: tuple[int, int, int] = (96, 128, 3),
    num_classes: int = 6,
    hidden_units: tuple[int, ...] = (512, 128),
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    name: str = "MLP",
) -> Any:
    """Constrói e compila um classificador Perceptron Multicamadas (MLP).

    Achata os tensores tridimensionais RGB e processa por camadas densas com ativação ReLU.

    Args:
        input_shape: Dimensão do tensor de entrada (altura, largura, canais).
        num_classes: Quantidade de categorias de saída.
        hidden_units: Tupla contendo o número de neurônios de cada camada oculta.
        dropout_rate: Taxa de descarte para regularização.
        learning_rate: Taxa de aprendizado do otimizador Adam.
        name: Identificador textual do modelo.

    Returns:
        Modelo tf.keras.Model compilado.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())

    for units in hidden_units:
        model.add(layers.Dense(units, activation="relu"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_cnn(
    input_shape: tuple[int | None, int | None, int] = (96, 128, 3),
    num_classes: int = 6,
    conv_filters: tuple[int, ...] = (32, 64, 128),
    dense_units: int = 128,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    name: str = "CNN",
) -> Any:
    """Constrói e compila uma Rede Neural Convolucional com GlobalAveragePooling2D.

    O emprego de GlobalAveragePooling2D viabiliza invariância dimensional estrita,
    permitindo que os mesmos pesos operem em resoluções espaciais distintas.

    Args:
        input_shape: Dimensão do tensor de entrada (altura, largura, canais).
        num_classes: Quantidade de categorias de saída.
        conv_filters: Tupla com o número de filtros de cada bloco convolucional.
        dense_units: Quantidade de neurônios na camada densa pós-pooling.
        dropout_rate: Taxa de descarte para regularização.
        learning_rate: Taxa de aprendizado do otimizador Adam.
        name: Identificador textual do modelo.

    Returns:
        Modelo tf.keras.Model compilado.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential(name=name)
    model.add(layers.Input(shape=input_shape))

    for filters in conv_filters:
        model.add(layers.Conv2D(filters, (3, 3), padding="same", activation="relu"))
        model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(dense_units, activation="relu"))

    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
