"""Distribution shift — aplica um shift de covariável controlado (escala +
offset numa feature) e mede a queda de acurácia e o drift de probabilidade
média. Modela "os dados de produção não são mais os de treino".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.robustness.perturbation import _proba


def _accuracy(model, X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> float:
    return float(np.mean((_proba(model, X) >= threshold).astype(int) == np.asarray(y)))


def covariate_shift(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    scales=(1.0, 1.25, 1.5, 2.0),
    offsets=(0.0,),
    threshold: float = 0.5,
) -> list[dict]:
    if feature not in X.columns:
        raise KeyError(feature)
    y = np.asarray(y)
    base_acc = _accuracy(model, X, y, threshold)
    base_mean_p = float(np.mean(_proba(model, X)))

    rows = []
    for s in scales:
        for o in offsets:
            Xs = X.copy()
            Xs[feature] = Xs[feature] * s + o
            acc = _accuracy(model, Xs, y, threshold)
            rows.append({
                "feature": feature, "scale": s, "offset": o,
                "accuracy": round(acc, 4),
                "accuracy_drop": round(base_acc - acc, 4),
                "mean_prob_drift": round(float(np.mean(_proba(model, Xs))) - base_mean_p, 4),
            })
    return rows


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre duas distribuições (treino vs shift)."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    quantiles = np.quantile(expected, np.linspace(0, 1, bins + 1))
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    e = np.histogram(expected, quantiles)[0] / len(expected) + 1e-6
    a = np.histogram(actual, quantiles)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))
