"""Transformer-Enhanced LSTM model definition (TensorFlow / Keras Functional API)."""
from tensorflow.keras import layers, models


def transformer_encoder(x, num_heads: int = 4, ff_dim: int = 128, dropout: float = 0.1):
    """A single Transformer encoder block: multi-head self-attention + feed-forward,
    each wrapped in a residual connection and layer normalization."""
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    attn = layers.Dropout(dropout)(attn)
    x1 = layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x, attn]))
    ff = layers.Dense(ff_dim, activation="relu")(x1)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    return layers.LayerNormalization(epsilon=1e-6)(layers.Add()([x1, ff]))


def build_transformer_lstm(n_features: int, n_classes: int):
    """Build the proposed Transformer-Enhanced LSTM classifier."""
    inp = layers.Input(shape=(1, n_features))
    x = layers.Dense(128, activation="relu")(inp)                        # feature embedding
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x) # BiLSTM 1
    x = layers.Bidirectional(layers.LSTM(64,  return_sequences=True))(x) # BiLSTM 2
    x = transformer_encoder(x, num_heads=4, ff_dim=128, dropout=0.2)     # attention block
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inp, out, name="Transformer_Enhanced_LSTM")


def build_simple_lstm(n_features: int, n_classes: int):
    """A plain LSTM baseline used for comparison."""
    return models.Sequential([
        layers.Input((1, n_features)),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])
