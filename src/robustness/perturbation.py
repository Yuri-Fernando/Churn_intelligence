"""Feature perturbation — mede quão instável é a fronteira de decisão do
modelo de churn a pequenas variações de features.

Se `tenure: 14 -> 15`, `tickets: 5 -> 4`, `usage: 120 -> 125` já derruba a
probabilidade de churn de 0.82 para 0.39, a fronteira é frágil.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _proba(model, X: pd.DataFrame) -> np.ndarray:
    p = model.predict_proba(X)
    return p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def feature_perturbation(
    model,
    X: pd.DataFrame,
    feature: str,
    deltas: list[float] | None = None,
    threshold: float = 0.5,
) -> list[dict]:
    """Para cada `delta`, soma `delta` à `feature` em todo o lote e mede:
    swing médio da probabilidade e fração de decisões que mudam de lado."""
    if feature not in X.columns:
        raise KeyError(feature)
    deltas = deltas or [-2, -1, -0.5, 0.5, 1, 2]
    base_p = _proba(model, X)
    base_decision = base_p >= threshold

    rows = []
    for d in deltas:
        Xp = X.copy()
        Xp[feature] = Xp[feature] + d
        p = _proba(model, Xp)
        rows.append({
            "delta": d,
            "mean_abs_prob_change": round(float(np.mean(np.abs(p - base_p))), 4),
            "decision_flip_rate": round(float(np.mean((p >= threshold) != base_decision)), 4),
        })
    return rows


def fragility_score(model, X: pd.DataFrame, features: list[str], threshold: float = 0.5) -> float:
    """Fração média de decisões que mudam com uma perturbação de ±1 unidade
    em cada feature listada — resumo único de fragilidade."""
    flips = []
    for f in features:
        for row in feature_perturbation(model, X, f, deltas=[-1, 1], threshold=threshold):
            flips.append(row["decision_flip_rate"])
    return round(float(np.mean(flips)) if flips else 0.0, 4)
