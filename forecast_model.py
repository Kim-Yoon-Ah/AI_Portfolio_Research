from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class ForecastModels:
    features: List[str]
    model_1d: RandomForestRegressor
    model_5d: RandomForestRegressor
    model_20d: RandomForestRegressor


def _train_single_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    # Simple time-series split for validation (no shuffling)
    tscv = TimeSeriesSplit(n_splits=3)
    best_model = None
    best_score = -np.inf

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_model = model

    return best_model if best_model is not None else RandomForestRegressor().fit(X, y)


def train_forecast_models(features_df: pd.DataFrame) -> ForecastModels | None:
    """
    Train global models for 1d, 5d, and 20d forward returns.
    """
    if features_df.empty:
        return None

    target_cols = ["future_return_1d", "future_return_5d", "future_return_20d"]
    feature_cols = [c for c in features_df.columns if c not in ("asset", "price", *target_cols)]

    X = features_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    y1 = features_df["future_return_1d"].copy()
    y5 = features_df["future_return_5d"].copy()
    y20 = features_df["future_return_20d"].copy()

    valid_mask = X.notna().all(axis=1) & y1.notna() & y5.notna() & y20.notna()
    X = X.loc[valid_mask]
    y1 = y1.loc[valid_mask]
    y5 = y5.loc[valid_mask]
    y20 = y20.loc[valid_mask]
    if X.empty:
        return None

    model_1d = _train_single_model(X, y1)
    model_5d = _train_single_model(X, y5)
    model_20d = _train_single_model(X, y20)

    return ForecastModels(features=feature_cols, model_1d=model_1d, model_5d=model_5d, model_20d=model_20d)


def _ensemble_distribution(model: RandomForestRegressor, x_row: np.ndarray) -> np.ndarray:
    # Collect predictions from all trees for one sample
    preds = np.array([estimator.predict(x_row.reshape(1, -1))[0] for estimator in model.estimators_])
    return preds


def forecast_for_asset(
    models: ForecastModels,
    latest_row: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Produce forecast statistics for a single asset from the trained models.

    Returns dict with keys:
      - '1d', '5d', '20d' each containing:
        - prob_positive
        - exp_return
        - low
        - high
    """
    x = latest_row[models.features].replace([np.inf, -np.inf], np.nan).values.astype(float)
    if not np.isfinite(x).all():
        return {
            "1d": {"prob_positive": 0.0, "exp_return": 0.0, "low": 0.0, "high": 0.0},
            "5d": {"prob_positive": 0.0, "exp_return": 0.0, "low": 0.0, "high": 0.0},
            "20d": {"prob_positive": 0.0, "exp_return": 0.0, "low": 0.0, "high": 0.0},
        }

    dist_1d = _ensemble_distribution(models.model_1d, x)
    dist_5d = _ensemble_distribution(models.model_5d, x)
    dist_20d = _ensemble_distribution(models.model_20d, x)

    def _stats(d: np.ndarray) -> Dict[str, float]:
        if d.size == 0:
            return {"prob_positive": 0.0, "exp_return": 0.0, "low": 0.0, "high": 0.0}
        return {
            "prob_positive": float((d > 0).mean()),
            "exp_return": float(d.mean()),
            "low": float(np.quantile(d, 0.05)),
            "high": float(np.quantile(d, 0.95)),
        }

    return {
        "1d": _stats(dist_1d),
        "5d": _stats(dist_5d),
        "20d": _stats(dist_20d),
    }

