"""Architectures. Modele propose : MS-Res-SE-BiGRU-Attention avec fusion RR.

Entree 1 : battement (L, 3) = precedent | courant | suivant  -> morphologie + contexte
Entree 2 : 6 descripteurs RR                                  -> rythme (classe S)
"""
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from . import config as C


# --------------------------------------------------------------------------- #
# Briques
# --------------------------------------------------------------------------- #
def se_block(x, ratio=8):
    ch = x.shape[-1]
    s = L.GlobalAveragePooling1D()(x)
    s = L.Dense(max(ch // ratio, 4), activation="relu")(s)
    s = L.Dense(ch, activation="sigmoid")(s)
    return L.Multiply()([x, L.Reshape((1, ch))(s)])


def res_block(x, filters, k=7, use_se=True, drop=0.1):
    shortcut = x
    if shortcut.shape[-1] != filters:
        shortcut = L.Conv1D(filters, 1, padding="same", use_bias=False)(shortcut)
    h = L.Conv1D(filters, k, padding="same", use_bias=False)(x)
    h = L.BatchNormalization()(h)
    h = L.Activation("relu")(h)
    h = L.Dropout(drop)(h)
    h = L.Conv1D(filters, k, padding="same", use_bias=False)(h)
    h = L.BatchNormalization()(h)
    if use_se:
        h = se_block(h)
    return L.Activation("relu")(L.Add()([h, shortcut]))


# keras.saving n'existe pas via le loader paresseux tf.keras de Colab :
# keras.utils.register_keras_serializable est l'API stable dans Keras 2 comme dans Keras 3.
_register = getattr(keras.utils, "register_keras_serializable",
                    getattr(getattr(keras, "saving", None), "register_keras_serializable", lambda **kw: (lambda c: c)))


@_register(package="ecg")
class AttentionPooling(L.Layer):
    """Pooling attentionnel ; expose les poids pour l'interpretabilite."""

    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.d1 = L.Dense(units, activation="tanh")
        self.d2 = L.Dense(1)

    def call(self, x):
        w = tf.nn.softmax(self.d2(self.d1(x)), axis=1)      # (B, T, 1)
        self.last_weights = w
        return tf.reduce_sum(x * w, axis=1)

    def get_config(self):
        return super().get_config()


# --------------------------------------------------------------------------- #
# Modele propose
# --------------------------------------------------------------------------- #
def build_proposed(use_rr=True, use_att=True, use_se=True, n_channels=C.N_CHANNELS,
                   name="PROPOSED"):
    sig = keras.Input((C.RESAMPLE_LEN, n_channels), name="signal")
    inputs = [sig]

    branches = [L.Conv1D(32, k, padding="same", activation="relu")(sig) for k in (3, 7, 15)]
    x = L.BatchNormalization()(L.Concatenate()(branches))          # stem multi-echelle

    x = res_block(x, 64, 7, use_se); x = L.MaxPooling1D(2)(x)
    x = res_block(x, 128, 5, use_se); x = L.MaxPooling1D(2)(x)
    x = res_block(x, 128, 3, use_se)

    x = L.Bidirectional(L.GRU(64, return_sequences=True))(x)

    if use_att:
        att = L.MultiHeadAttention(num_heads=4, key_dim=32, name="mha")(x, x)
        x = L.LayerNormalization()(L.Add()([x, att]))
        z = AttentionPooling(name="att_pool")(x)
    else:
        z = L.GlobalAveragePooling1D()(x)

    if use_rr:
        rr = keras.Input((C.RR_DIM,), name="rr")
        inputs.append(rr)
        r = L.BatchNormalization()(L.Dense(32, activation="relu")(rr))
        z = L.Concatenate()([z, r])

    z = L.Dropout(0.4)(L.Dense(128, activation="relu")(z))
    z = L.Dropout(0.3)(L.Dense(64, activation="relu")(z))
    logits = L.Dense(C.N_CLASSES, name="logits")(z)
    out = L.Activation("softmax", name="proba", dtype="float32")(logits)
    return keras.Model(inputs, out, name=name)


# --------------------------------------------------------------------------- #
# Baselines de la litterature (morphologie seule)
# --------------------------------------------------------------------------- #
def build_cnn1d(n_channels=C.N_CHANNELS):
    sig = keras.Input((C.RESAMPLE_LEN, n_channels), name="signal")
    x = sig
    for f, k in [(64, 7), (128, 5), (128, 3)]:
        x = L.Conv1D(f, k, padding="same")(x)
        x = L.Activation("relu")(L.BatchNormalization()(x))
        x = L.MaxPooling1D(2)(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(L.Dense(64, activation="relu")(x))
    return keras.Model(sig, L.Dense(C.N_CLASSES, activation="softmax")(x), name="CNN1D")


def build_bilstm(n_channels=C.N_CHANNELS):
    sig = keras.Input((C.RESAMPLE_LEN, n_channels), name="signal")
    x = L.Dropout(0.3)(L.Bidirectional(L.LSTM(64, return_sequences=True))(sig))
    x = L.Dropout(0.3)(L.Bidirectional(L.LSTM(32))(x))
    x = L.BatchNormalization()(L.Dense(32, activation="relu")(x))
    return keras.Model(sig, L.Dense(C.N_CLASSES, activation="softmax")(x), name="BiLSTM")


def build_cnn_bilstm(n_channels=C.N_CHANNELS):
    sig = keras.Input((C.RESAMPLE_LEN, n_channels), name="signal")
    x = L.MaxPooling1D(2)(L.Conv1D(64, 7, padding="same", activation="relu")(sig))
    x = L.MaxPooling1D(2)(L.Conv1D(128, 5, padding="same", activation="relu")(x))
    x = L.Dropout(0.3)(L.Bidirectional(L.LSTM(64))(x))
    x = L.Dense(64, activation="relu")(x)
    return keras.Model(sig, L.Dense(C.N_CLASSES, activation="softmax")(x), name="CNN_BiLSTM")


def build_resnet1d(n_channels=C.N_CHANNELS):
    sig = keras.Input((C.RESAMPLE_LEN, n_channels), name="signal")
    x = L.Conv1D(64, 7, padding="same", activation="relu")(sig)
    for f in (64, 128, 128):
        x = L.MaxPooling1D(2)(res_block(x, f, 5, use_se=False))
    x = L.Dense(64, activation="relu")(L.GlobalAveragePooling1D()(x))
    return keras.Model(sig, L.Dense(C.N_CLASSES, activation="softmax")(x), name="ResNet1D")


BUILDERS = {
    "CNN1D": build_cnn1d,
    "BiLSTM": build_bilstm,
    "CNN_BiLSTM": build_cnn_bilstm,
    "ResNet1D": build_resnet1d,
    "PROPOSED": lambda: build_proposed(),
    "ABL_noRR": lambda: build_proposed(use_rr=False, name="ABL_noRR"),
    "ABL_noCTX": lambda: build_proposed(n_channels=1, name="ABL_noCTX"),
    "ABL_noATT": lambda: build_proposed(use_att=False, name="ABL_noATT"),
    "ABL_noSE": lambda: build_proposed(use_se=False, name="ABL_noSE"),
    "ABL_CE": lambda: build_proposed(name="ABL_CE"),        # meme reseau, perte differente
    "ABL_noAUG": lambda: build_proposed(name="ABL_noAUG"),  # meme reseau, sans augmentation
}


def model_inputs(model, X, R):
    """Adapte l'alimentation : baselines = signal seul, modele propose = signal + RR."""
    names = [t.name.split(":")[0].split("/")[0] for t in model.inputs]
    if len(model.inputs) == 1:
        Xin = X[..., 1:2] if model.input_shape[-1] == 1 else X
        return Xin
    Xin = X[..., 1:2] if model.input_shape[0][-1] == 1 else X
    return {"signal": Xin, "rr": R}
