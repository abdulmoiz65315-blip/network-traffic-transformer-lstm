"""Train the Transformer-Enhanced LSTM on a network-traffic CSV/parquet file.

Usage:
    python train.py --data ../cicdarknet2020.parquet --target Label --epochs 40
"""
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

from preprocessing import load_any, preprocess, find_label_columns
from model import build_transformer_lstm

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to .parquet or .csv")
    ap.add_argument("--target", default=None, help="label column (auto-detected if omitted)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--out", default="../results/transformer_lstm.keras")
    args = ap.parse_args()

    np.random.seed(SEED); tf.random.set_seed(SEED)

    df = load_any(args.data)
    target = args.target or find_label_columns(df)[0]
    print("Target column:", target)

    data = preprocess(df, target)
    nf, ncl = data["n_features"], data["n_classes"]
    Xtr = data["X_train"].reshape(-1, 1, nf)
    Xva = data["X_val"].reshape(-1, 1, nf)

    model = build_transformer_lstm(nf, ncl)
    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
        callbacks.ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True),
    ]
    model.fit(Xtr, data["y_train"], validation_data=(Xva, data["y_val"]),
              epochs=args.epochs, batch_size=args.batch_size, callbacks=cbs, verbose=2)
    model.save(args.out)
    print("Saved model to", args.out)


if __name__ == "__main__":
    main()
