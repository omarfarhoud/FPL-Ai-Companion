"""
Inference utilities for the project.

This module provides simple functions to:
- train_and_save_linear_model(dataset_path, model_path, features_path)
- load_model_and_features(model_path, features_path)
- predict_from_raw(raw_row: dict | pd.DataFrame, model_path, features_path)

The inference function expects a preprocessed row shaped like the training data's features
(e.g., same dummy columns from `position` and `team_x`). For convenience it will attempt
basic feature engineering covered in `training_prep.ipynb`:
- ensure `was_home` is integer
- create dummies for `position` and `team_x` using the saved feature list (missing dummies -> 0)

If no saved model exists the training function will fit a LinearRegression on
`final_prepared_dataset.csv` and persist the model and features to disk.

Note: For production use you should replace this with a stable preprocessing pipeline
(e.g., sklearn Pipeline or saved preprocessing artifact). This implementation is
intended to be a minimal, reproducible inference helper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_and_save_linear_model(dataset_path: Union[str, Path], model_path: Union[str, Path], features_path: Union[str, Path]):
    """Train a LinearRegression on the final prepared dataset and save model + feature list.

    Args:
        dataset_path: path to final_prepared_dataset.csv
        model_path: where to write the trained model (.joblib)
        features_path: where to write the feature column list (.json)
    """
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    features_path = Path(features_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)

    # assume training_prep selected these drops
    X = df.drop(columns=[c for c in ["name", "GW", "upcoming_total_points"] if c in df.columns])
    y = df["upcoming_total_points"]

    model = LinearRegression()
    model.fit(X, y)

    # Save model and feature list
    model_path.parent.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    features = list(X.columns)
    features_path.write_text(json.dumps(features, indent=2))

    return model, features


def load_model_and_features(model_path: Union[str, Path], features_path: Union[str, Path]):
    model_path = Path(model_path)
    features_path = Path(features_path)
    if not model_path.exists() or not features_path.exists():
        raise FileNotFoundError("Model or features file not found. Run train_and_save_linear_model first.")
    model = joblib.load(model_path)
    features = json.loads(features_path.read_text())
    return model, features


def _preprocess_raw_row(raw: Union[Dict, pd.DataFrame], features: List[str]) -> pd.DataFrame:
    """Convert a raw input (dict or single-row DataFrame) into a DataFrame with columns matching features.

    This function attempts basic steps used in `training_prep.ipynb`:
    - ensures was_home as int
    - creates dummies for `position` and `team_x` columns that appeared in training
    - fills missing numeric columns with 0 and missing dummies with 0

    Returns a single-row DataFrame with columns ordered as `features`.
    """
    if isinstance(raw, dict):
        row = pd.DataFrame([raw])
    elif isinstance(raw, pd.DataFrame):
        if len(raw) != 1:
            raise ValueError("raw DataFrame must be a single row")
        row = raw.copy()
    else:
        raise TypeError("raw must be dict or single-row DataFrame")

    # Basic conversions
    if "was_home" in row.columns:
        try:
            row["was_home"] = row["was_home"].astype(int)
        except Exception:
            row["was_home"] = row["was_home"].apply(lambda x: 1 if str(x).lower() in ("1", "true", "yes", "home") else 0)

    # categorical dummies used during training (position_, team_x_...)
    # find categorical prefixes in feature list
    cat_prefixes = set()
    for f in features:
        if f.startswith("position_"):
            cat_prefixes.add("position")
        if f.startswith("team_x_"):
            cat_prefixes.add("team_x")

    # create dummies for any categorical fields present in the raw input
    for prefix in cat_prefixes:
        col = prefix
        if col in row.columns:
            dummies = pd.get_dummies(row[col].astype(str), prefix=prefix)
            # ensure columns match training dummies (will align later)
            row = pd.concat([row.drop(columns=[col]), dummies], axis=1)

    # Ensure all feature columns exist; fill missing with 0
    out = pd.DataFrame(columns=features)
    single = pd.Series(index=features, dtype=object)
    for f in features:
        if f in row.columns:
            single[f] = row.iloc[0][f]
        else:
            single[f] = 0

    # Convert numeric-like columns to numeric where possible (avoid errors='ignore' deprecation)
    for col in single.index:
        val = single[col]
        try:
            num = pd.to_numeric(val)
            single[col] = num
        except Exception:
            # leave as-is (likely categorical/string)
            single[col] = val
    return single.to_frame().T


def predict_from_raw(raw_input: Union[Dict, pd.DataFrame], model_path: Union[str, Path] = "./models/linear_model.joblib", features_path: Union[str, Path] = "./models/features.json") -> float:
    """Load model/features and return prediction for a single raw input row.

    If the model or features file are not present, this function raises an error; call train_and_save_linear_model first.
    """
    model, features = load_model_and_features(model_path, features_path)
    X_row = _preprocess_raw_row(raw_input, features)
    pred = model.predict(X_row)
    return float(pred[0])


def main():
    """Simple CLI test: train model if missing, then predict for a sample row from the final prepared dataset.

    Usage (from repo root):
        python src/models/inference.py
    """
    from pathlib import Path
    import sys

    repo_src = Path(__file__).resolve().parents[1]
    dataset_path = repo_src / 'data' / 'cleaned' / 'final_prepared_dataset.csv'
    model_path = Path(__file__).parent / 'linear_model.joblib'
    features_path = Path(__file__).parent / 'features.json'

    print('Dataset path:', dataset_path)
    if not dataset_path.exists():
        print('Error: prepared dataset not found at', dataset_path)
        print('Please create the dataset (see training_prep.ipynb) and try again.')
        sys.exit(2)

    # Train model if necessary
    if not model_path.exists() or not features_path.exists():
        print('Model or features not found — training a new LinearRegression model...')
        model, features = train_and_save_linear_model(dataset_path, model_path, features_path)
        print('Trained and saved model to', model_path)
    else:
        print('Loading existing model from', model_path)
        model, features = load_model_and_features(model_path, features_path)

    # Load a sample row (use first row without the target)
    df = pd.read_csv(dataset_path)
    if 'upcoming_total_points' in df.columns:
        sample = df.drop(columns=['upcoming_total_points']).iloc[[0]]
    else:
        sample = df.iloc[[0]]

    print('\nSample input (first row):')
    print(sample.head(1).to_dict(orient='records')[0])

    pred = predict_from_raw(sample, model_path=model_path, features_path=features_path)
    print(f"\nPrediction for sample row: {pred:.4f}")


if __name__ == '__main__':
    main()
