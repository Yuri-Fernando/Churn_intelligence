import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.robustness import (
    OODDetector,
    covariate_shift,
    feature_perturbation,
    fragility_score,
    missing_data_impact,
    missing_data_sweep,
    psi,
    robustness_report,
)


@pytest.fixture(scope="module")
def model_and_data():
    rng = np.random.default_rng(0)
    n = 800
    df = pd.DataFrame({
        "recency_days": rng.integers(1, 180, n).astype(float),
        "frequency": rng.integers(0, 30, n).astype(float),
        "avg_session_duration": rng.uniform(1, 20, n),
        "intensity": rng.uniform(1, 10, n),
        "tickets": rng.integers(0, 8, n).astype(float),
        "tenure": rng.integers(1, 60, n).astype(float),
    })
    logit = 0.02 * df["recency_days"] - 0.06 * df["frequency"] + 0.25 * df["tickets"] - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    split = int(0.7 * n)
    X_tr, X_te = df.iloc[:split], df.iloc[split:]
    y_tr, y_te = y[:split], y[split:]
    model = RandomForestClassifier(n_estimators=60, random_state=0).fit(X_tr, y_tr)
    return model, X_tr, X_te, y_te


def test_feature_perturbation_reports_flip_rates(model_and_data):
    model, _, X_te, _ = model_and_data
    rows = feature_perturbation(model, X_te, "tickets", deltas=[-2, -1, 1, 2])
    assert len(rows) == 4
    assert all(0.0 <= r["decision_flip_rate"] <= 1.0 for r in rows)
    # perturbação maior tende a causar >= flips que a menor
    assert rows[-1]["mean_abs_prob_change"] >= rows[2]["mean_abs_prob_change"] - 1e-6


def test_fragility_score_in_unit_interval(model_and_data):
    model, _, X_te, _ = model_and_data
    s = fragility_score(model, X_te, ["recency_days", "frequency", "tickets"])
    assert 0.0 <= s <= 1.0


def test_missing_data_impact_and_sweep(model_and_data):
    model, _, X_te, _ = model_and_data
    r = missing_data_impact(model, X_te, ["frequency"], strategy="median")
    assert r["decision_flip_rate"] >= 0.0
    sweep = missing_data_sweep(model, X_te, ["frequency", "tickets", "recency_days"])
    assert sweep == sorted(sweep, key=lambda d: -d["decision_flip_rate"])


def test_covariate_shift_degrades_accuracy(model_and_data):
    model, _, X_te, y_te = model_and_data
    rows = covariate_shift(model, X_te, y_te, "recency_days", scales=(1.0, 2.0, 3.0))
    assert rows[0]["accuracy_drop"] == 0.0
    assert rows[-1]["accuracy_drop"] >= -1e-9  # shift não deve melhorar acurácia


def test_psi_zero_for_same_distribution(model_and_data):
    _, X_tr, _, _ = model_and_data
    v = X_tr["frequency"].to_numpy()
    assert psi(v, v) < 0.01


def test_ood_detector_flags_out_of_range_rows(model_and_data):
    model, X_tr, X_te, _ = model_and_data
    det = OODDetector().fit(X_tr)
    assert det.flag_rate(X_tr) <= 0.15  # poucos flags no próprio treino
    X_ood = X_te.copy()
    X_ood["recency_days"] = X_ood["recency_days"] * 100 + 10_000
    assert det.flag_rate(X_ood) > det.flag_rate(X_te)


def test_robustness_report_renders_and_classifies_risk(model_and_data):
    model, X_tr, X_te, y_te = model_and_data
    rep = robustness_report(model, X_tr, X_te, y_te,
                            key_features=["recency_days", "frequency", "tickets", "tenure"])
    assert rep["robustness_risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert "MODEL ROBUSTNESS REPORT" in rep["rendered"]
    assert rep["recommendations"]
