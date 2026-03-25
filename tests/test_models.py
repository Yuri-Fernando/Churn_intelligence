import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.churn_model import train_models
from src.models.model_utils import split_features_target, evaluate_model
from src.models.clustering_model import cluster_users, cluster_summary


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "recency_days":         np.random.uniform(1, 180, n),
        "frequency":            np.random.uniform(0, 30, n),
        "avg_session_duration": np.random.uniform(1, 20, n),
        "intensity":            np.random.uniform(1, 10, n),
        "engagement_trend":     np.zeros(n),
        "churn":                np.random.randint(0, 2, n),
    })


def test_train_models_returns_dict(sample_df):
    models, results = train_models(sample_df)
    assert isinstance(models, dict)
    assert isinstance(results, dict)
    assert "RandomForest" in models
    assert "LogisticRegression" in models


def test_train_models_metrics_keys(sample_df):
    _, results = train_models(sample_df)
    for name, metrics in results.items():
        for key in ["roc_auc", "precision", "recall"]:
            assert key in metrics, f"{name} sem metrica {key}"


def test_evaluate_model_range(sample_df):
    models, _ = train_models(sample_df)
    X_train, X_test, y_train, y_test = split_features_target(sample_df)
    metrics = evaluate_model(models["RandomForest"], X_test, y_test)
    assert 0.0 <= metrics["roc_auc"]  <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"]    <= 1.0


def test_cluster_users_adds_columns(sample_df):
    result = cluster_users(sample_df, n_clusters=4)
    assert "cluster_id" in result.columns
    assert "cluster_label" in result.columns


def test_cluster_users_valid_labels(sample_df):
    result = cluster_users(sample_df, n_clusters=4)
    valid = {"engaged", "occasional", "at_risk", "inactive", "unknown"}
    assert set(result["cluster_label"].unique()).issubset(valid)


def test_cluster_summary_returns_dataframe(sample_df):
    result = cluster_users(sample_df, n_clusters=4)
    summary = cluster_summary(result)
    assert isinstance(summary, pd.DataFrame)
