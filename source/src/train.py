"""Boucle d'entrainement : multi-seeds, augmentation, adaptation patient, ensemble."""
from __future__ import annotations
import os, random, time
import numpy as np
import tensorflow as tf
from tensorflow import keras

from . import config as C
from .models import BUILDERS, model_inputs
from .losses import focal_loss, weighted_cross_entropy
from .preprocessing import augment_batch


def set_all_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


class AugmentedSequence(keras.utils.Sequence):
    """Augmentation a la volee, TRAIN uniquement, reproductible."""

    def __init__(self, X, R, y_cat, batch_size, seed=0, augment=True, with_rr=True,
                 single_channel=False):
        super().__init__()
        self.X, self.R, self.y = X, R, y_cat
        self.bs, self.augment, self.with_rr = batch_size, augment, with_rr
        self.single_channel = single_channel
        self.rng = np.random.default_rng(seed)
        self.idx = np.arange(len(y_cat))
        self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.bs))

    def __getitem__(self, i):
        s = self.idx[i * self.bs:(i + 1) * self.bs]
        xb = self.X[s]
        if self.augment:
            xb = augment_batch(xb, self.rng)
        if self.single_channel:
            xb = xb[..., 1:2]
        yb = self.y[s]
        return ({"signal": xb, "rr": self.R[s]} if self.with_rr else xb), yb

    def on_epoch_end(self):
        self.rng.shuffle(self.idx)


def _compile(model, model_name, class_weights):
    loss = weighted_cross_entropy(class_weights) if model_name == "ABL_CE" \
        else focal_loss(C.FOCAL_GAMMA, class_weights)
    try:
        opt = keras.optimizers.AdamW(learning_rate=C.LR, weight_decay=C.WEIGHT_DECAY)
    except Exception:
        opt = keras.optimizers.Adam(learning_rate=C.LR)
    # Keras 3 ne sait pas deduire la bonne metrique depuis une perte personnalisee :
    # "accuracy" tombe alors sur keras.metrics.Accuracy (comparaison brute) et affiche ~0.
    # On impose donc explicitement l'exactitude categorielle.
    model.compile(optimizer=opt, loss=loss,
                  metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")])
    return model


def train_model(model_name, split, seed, epochs=None, verbose=2):
    """Entraine un modele sur TRAIN, selectionne sur VALIDATION. Le test n'est jamais vu."""
    set_all_seeds(seed)
    keras.backend.clear_session()
    model = BUILDERS[model_name]()
    _compile(model, model_name, split.class_weights)

    with_rr = len(model.inputs) == 2
    single = (model.input_shape[0][-1] if with_rr else model.input_shape[-1]) == 1
    augment = model_name != "ABL_noAUG"

    ytr_cat = keras.utils.to_categorical(split.ytr, C.N_CLASSES)
    yva_cat = keras.utils.to_categorical(split.yva, C.N_CLASSES)
    train_seq = AugmentedSequence(split.Xtr, split.Rtr, ytr_cat, C.BATCH_SIZE, seed,
                                  augment=augment, with_rr=with_rr, single_channel=single)
    val_data = (model_inputs(model, split.Xva, split.Rva), yva_cat)

    ckpt = C.MOD_DIR / f"{model_name}_seed{seed}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping("val_loss", patience=C.PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau("val_loss", factor=0.5,
                                          patience=max(3, C.PATIENCE // 3), min_lr=1e-6),
        keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", save_best_only=True),
    ]
    t0 = time.time()
    history = model.fit(train_seq, validation_data=val_data,
                        epochs=epochs or C.EPOCHS, callbacks=callbacks, verbose=verbose)
    train_time = time.time() - t0
    return model, history.history, train_time


def adapt_to_patients(model, split, seed, epochs=C.ADAPT_EPOCHS, lr=C.ADAPT_LR,
                      model_name=None):
    """Reglage fin sur les 5 premieres minutes de CHAQUE enregistrement de test.

    Protocole AAMI patient-adaptatif : ces battements sont exclus du test.
    """
    if len(split.yad) == 0:
        return model
    set_all_seeds(seed + 500)
    # clone_model() tente de deserialiser la perte personnalisee -> TypeError.
    # On reconstruit donc l'architecture a neuf puis on copie les poids.
    name = model_name or model.name
    if name in BUILDERS:
        adapted = BUILDERS[name]()
    else:
        adapted = model.__class__.from_config(model.get_config())
    adapted.set_weights(model.get_weights())
    _compile(adapted, "ADAPT", split.class_weights)
    adapted.optimizer.learning_rate.assign(lr)
    yad_cat = keras.utils.to_categorical(split.yad, C.N_CLASSES)
    adapted.fit(model_inputs(adapted, split.Xad, split.Rad), yad_cat,
                epochs=epochs, batch_size=64, verbose=0, shuffle=True)
    return adapted


def predict_proba(model, X, R, batch_size=512, tta=0, seed=0):
    """Prediction, avec augmentation au test (TTA) optionnelle."""
    p = np.asarray(model.predict(model_inputs(model, X, R), batch_size=batch_size, verbose=0))
    if tta > 0:
        rng = np.random.default_rng(seed)
        for _ in range(tta):
            Xa = augment_batch(X, rng, p=1.0, amp=(0.95, 1.05), noise_sd=0.01,
                               shift_max=2, warp=0.02)
            p += np.asarray(model.predict(model_inputs(model, Xa, R),
                                          batch_size=batch_size, verbose=0))
        p /= (tta + 1)
    return p.astype(np.float32)


def get_logits_model(model):
    try:
        return keras.Model(model.inputs, model.get_layer("logits").output)
    except Exception:
        return None
