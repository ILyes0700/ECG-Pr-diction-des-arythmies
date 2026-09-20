"""Fonctions de cout adaptees au desequilibre extreme des classes ECG."""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from . import config as C


def _weight_vector(class_weights):
    if class_weights is None:
        return tf.ones((C.N_CLASSES,), tf.float32)
    return tf.constant([class_weights[i] for i in range(C.N_CLASSES)], dtype=tf.float32)


def focal_loss(gamma=C.FOCAL_GAMMA, class_weights=None, label_smoothing=C.LABEL_SMOOTHING):
    w = _weight_vector(class_weights)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing > 0:
            y_true = y_true * (1.0 - label_smoothing) + label_smoothing / C.N_CLASSES
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        return tf.reduce_sum(tf.pow(1.0 - y_pred, gamma) * ce * w, axis=-1)

    return loss_fn


def weighted_cross_entropy(class_weights=None):
    w = _weight_vector(class_weights)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        return tf.reduce_sum(-y_true * tf.math.log(y_pred) * w, axis=-1)

    return loss_fn


def cost_sensitive_loss(cost_matrix=None, class_weights=None):
    """Penalise davantage les confusions cliniquement graves (V manque, S manque)."""
    cost = tf.constant(C.COST_MATRIX if cost_matrix is None else cost_matrix, tf.float32)
    w = _weight_vector(class_weights)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        expected_cost = tf.reduce_sum(tf.matmul(y_true, cost) * y_pred, axis=-1)
        ce = tf.reduce_sum(-y_true * tf.math.log(y_pred) * w, axis=-1)
        return ce + 0.1 * expected_cost

    return loss_fn
