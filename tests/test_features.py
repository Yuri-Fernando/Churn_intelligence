import pytest
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.build_features import build_features


@pytest.fixture
def raw_csv(tmp_path):
    """Cria CSV sintetico para testes."""
    df = pd.DataFrame({
        "Customer_ID":               [f"c{i}" for i in range(100)],
        "Days_Since_Last_Purchase":  np.random.randint(1, 180, 100),
        "Login_Frequency":           np.random.randint(0, 30, 100),
        "Session_Duration_Avg":      np.random.uniform(1, 20, 100),
        "Pages_Per_Session":         np.random.uniform(1, 10, 100),
        "Gender":                    np.random.choice(["M", "F"], 100),
        "Country":                   np.random.choice(["BR", "US", "UK"], 100),
        "City":                      np.random.choice(["SP", "RJ", "BH"], 100),
        "Payment_Method_Diversity":  np.random.choice(["Low", "Medium", "High"], 100),
        "Signup_Quarter":            np.random.choice(["Q1", "Q2", "Q3", "Q4"], 100),
        "Churned":                   np.random.randint(0, 2, 100),
    })
    path = str(tmp_path / "raw.csv")
    out  = str(tmp_path / "features.csv")
    df.to_csv(path, index=False)
    return path, out


def test_build_features_returns_dataframe(raw_csv):
    path, out = raw_csv
    df = build_features(input_csv=path, output_csv=out)
    assert isinstance(df, pd.DataFrame)


def test_build_features_has_required_columns(raw_csv):
    path, out = raw_csv
    df = build_features(input_csv=path, output_csv=out)
    for col in ["recency_days", "frequency", "avg_session_duration", "intensity", "churn"]:
        assert col in df.columns, f"Coluna ausente: {col}"


def test_build_features_no_nulls(raw_csv):
    path, out = raw_csv
    df = build_features(input_csv=path, output_csv=out)
    assert df.isnull().sum().sum() == 0, "DataFrame contem valores nulos"


def test_build_features_churn_is_binary(raw_csv):
    path, out = raw_csv
    df = build_features(input_csv=path, output_csv=out)
    assert set(df["churn"].unique()).issubset({0, 1})


def test_build_features_saves_csv(raw_csv):
    path, out = raw_csv
    build_features(input_csv=path, output_csv=out)
    assert os.path.exists(out)
