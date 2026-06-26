"""Evaluate a trained Transformer-Enhanced LSTM model.

Usage:
    python evaluate.py --data ../cicdarknet2020.parquet --model ../results/transformer_lstm.keras
"""
import argparse
import tensorflow as tf
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

from preprocessing import load_any, preprocess, find_label_columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default=None)
    args = ap.parse_args()

    df = load_any(args.data)
    target = args.target or find_label_columns(df)[0]
    data = preprocess(df, target, use_smote=False)
    nf = data["n_features"]

    model = tf.keras.models.load_model(args.model)
    proba = model.predict(data["X_test"].reshape(-1, 1, nf), verbose=0)
    pred = proba.argmax(1)
    yte = data["y_test"]

    print("Accuracy :", round(accuracy_score(yte, pred), 4))
    print("Precision:", round(precision_score(yte, pred, average="weighted", zero_division=0), 4))
    print("Recall   :", round(recall_score(yte, pred, average="weighted", zero_division=0), 4))
    print("F1       :", round(f1_score(yte, pred, average="weighted", zero_division=0), 4))
    print("\nClassification report:\n",
          classification_report(yte, pred, target_names=data["class_names"], zero_division=0))
    print("Confusion matrix:\n", confusion_matrix(yte, pred))


if __name__ == "__main__":
    main()
