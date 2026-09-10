"""Missing-data attacks — simula features ausentes em inferência (falha de
ingestão, campo não preenchido) e mede o quanto a predição muda em relação
ao caso completo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.robustness.perturbation import _proba


def missing_data_impact(
    model,
    X: pd.DataFrame,
    features: list[str],
    strategy: str = "median",
    fill_values: dict | None = None,
    threshold: float = 0.5,
) -> dict:
    """Zera/imputa `features` e compara com a predição do lote completo.

    `strategy`: "median" | "mean" | "zero" | "given" (usa `fill_values`)."""
    base_p = _proba(model, X)
    Xm = X.copy()

    for f in features:
        if f not in Xm.columns:
            raise KeyError(f)
        if strategy == "zero":
            val = 0.0
        elif strategy == "mean":
            val = float(X[f].mean())
        elif strategy == "given":
            val = float((fill_values or {}).get(f, 0.0))
        else:  # median
            val = float(X[f].median())
        Xm[f] = val

    p = _proba(model, Xm)
    return {
        "features_missing": features,
        "strategy": strategy,
        "mean_abs_prob_change": round(float(np.mean(np.abs(p - base_p))), 4),
        "decision_flip_rate": round(float(np.mean((p >= threshold) != (base_p >= threshold))), 4),
        "max_abs_prob_change": round(float(np.max(np.abs(p - base_p))), 4),
    }


def missing_data_sweep(model, X: pd.DataFrame, features: list[str], threshold: float = 0.5) -> list[dict]:
    """Impacto de perder cada feature individualmente (ranking de dependência)."""
    out = []
    for f in features:
        r = missing_data_impact(model, X, [f], strategy="median", threshold=threshold)
        out.append({"feature": f, **{k: r[k] for k in ("mean_abs_prob_change", "decision_flip_rate")}})
    return sorted(out, key=lambda d: -d["decision_flip_rate"])
