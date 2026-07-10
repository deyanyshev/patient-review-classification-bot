import tensorflow as tf
from tensorflow.keras import layers, models


@tf.keras.utils.register_keras_serializable(package="triage_nn")
class PaddingMask(layers.Layer):
    """Возвращает float-маску (1.0 = реальный токен, 0.0 = паддинг)."""

    def call(self, tokens):
        return tf.cast(tf.not_equal(tokens, 0), tf.float32)


@tf.keras.utils.register_keras_serializable(package="triage_nn")
class AttentionPooling(layers.Layer):
    """
    Attention pooling по временной оси с учётом маски паддинга.
    Вход: hidden_states (batch, seq_len, dim), mask (batch, seq_len)
    Выход: (batch, dim)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1)

    def call(self, inputs):
        hidden_states, mask = inputs

        scores = self.score_dense(hidden_states)
        scores = tf.squeeze(scores, axis=-1)

        scores = scores + (1.0 - mask) * -1e9

        weights = tf.nn.softmax(scores, axis=1)
        weights = tf.expand_dims(weights, axis=-1)

        pooled = tf.reduce_sum(hidden_states * weights, axis=1)
        return pooled


def build_model(
    vocab_size: int,
    embed_dim: int = 64,
    hidden_dim: int = 64,
    max_len: int = 64,
    num_classes: int = 3,
    dropout: float = 0.5,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None

    inputs = layers.Input(shape=(max_len,), dtype="int32", name="tokens")

    mask = PaddingMask(name="padding_mask")(inputs)

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        mask_zero=True,
        embeddings_regularizer=reg,
        name="embedding",
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            kernel_regularizer=reg,
            recurrent_regularizer=reg,
        ),
        name="bilstm",
    )(x)

    pooled = AttentionPooling(name="attention_pooling")([x, mask])

    x = layers.Dropout(dropout)(pooled)
    x = layers.Dense(hidden_dim, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="triage_classifier")
    return model
