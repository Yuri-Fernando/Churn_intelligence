"""Detecção de entradas fora de distribuição (OOD).

Calibra, num conjunto de referência (treino), limites por feature (quantis
1%/99%) e a média/desvio para z-score. Em inferência, sinaliza linhas com
features fora dos limites ou com z-score agregado alto — candidatas a
predição não-confiável.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class OODDetector:
    def __init__(self, q_low: float = 0.01, q_high: float = 0.99, z_threshold: float = 4.0):
        self.q_low = q_low
        self.q_high = q_high
        self.z_threshold = z_threshold
        self.lo_: pd.Series | None = None
        self.hi_: pd.Series | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, X: pd.DataFrame) -> "OODDetector":
        num = X.select_dtypes(include=[np.number])
        self.columns_ = list(num.columns)
        self.lo_ = num.quantile(self.q_low)
        self.hi_ = num.quantile(self.q_high)
        self.mean_ = num.mean()
        self.std_ = num.std().replace(0, 1.0)
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        num = X[self.columns_].astype(float)
        out_of_range = ((num < self.lo_) | (num > self.hi_)).sum(axis=1)
        max_z = ((num - self.mean_) / self.std_).abs().max(axis=1)
        return pd.DataFrame({"features_out_of_range": out_of_range, "max_z": max_z}, index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = OOD (não-confiável), 0 = dentro da distribuição de referência."""
        s = self.score(X)
        return ((s["features_out_of_range"] > 0) | (s["max_z"] > self.z_threshold)).astype(int).to_numpy()

    def flag_rate(self, X: pd.DataFrame) -> float:
        return round(float(np.mean(self.predict(X))), 4)
