"""Preprocessing utilities for network-traffic classification.

Mirrors the cleaning / encoding / splitting / scaling / SMOTE pipeline used in the
notebook so the same steps can be run from scripts.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

SEED = 42
DROP_LIKE = ["Flow ID", "Src IP", "Source IP", "Dst IP", "Destination IP",
             "Timestamp", "Fwd Header Length.1"]


def load_any(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_label_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.lower().startswith("label")]


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))
    return X


def preprocess(df: pd.DataFrame, target: str,
               test_size: float = 0.2, val_size: float = 0.1, use_smote: bool = True):
    """Return scaled train/val/test splits plus metadata."""
    label_cols = find_label_columns(df)
    work = df.drop_duplicates().reset_index(drop=True)
    drop_cols = [c for c in work.columns if c in DROP_LIKE] + \
                [c for c in label_cols if c != target]
    work = work.drop(columns=[c for c in drop_cols if c in work.columns], errors="ignore")

    y_raw = work[target].astype(str).str.strip()
    X = clean_features(work.drop(columns=[target]))

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X.values, y, test_size=test_size, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=SEED)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            X_tr, y_tr = SMOTE(random_state=SEED, k_neighbors=3).fit_resample(X_tr, y_tr)
        except Exception as exc:  # noqa
            print("SMOTE skipped:", exc)

    return {
        "X_train": X_tr, "X_val": X_va, "X_test": X_te,
        "y_train": y_tr, "y_val": y_va, "y_test": y_te,
        "class_names": list(le.classes_),
        "n_features": X_tr.shape[1], "n_classes": len(le.classes_),
        "scaler": scaler, "label_encoder": le,
    }
