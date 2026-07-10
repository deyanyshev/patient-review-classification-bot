import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from data import load_dataset, save_vocab, MAX_LEN
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU доступен:", bool(gpus), gpus)

    X_train, y_train, vocab = load_dataset('data/train.csv')
    X_val, y_val, _ = load_dataset('data/val.csv', vocab=vocab)

    save_vocab(vocab, os.path.join(args.out_dir, "vocab.json"))

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
    print("Веса классов (0,1,2 = метки 1,2,3):", class_weight)

    model = build_model(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_len=MAX_LEN,
        num_classes=3,
    )
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = os.path.join(args.out_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = model.predict(X_val, batch_size=args.batch_size).argmax(axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")
    print("\nVal macro-F1:", f1)
    print(classification_report(y_val, y_pred, target_names=["1_минимальное", "2_проблема", "3_ЧП"]))

    print(f"\nЛучшая модель сохранена в {ckpt_path}")


if __name__ == "__main__":
    main()
