import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.recommendations import recommend_action, score_to_segment, score_to_risk_level
from src.online_prediction.feature_builder import build_online_features


# ---- feature_builder ----

def test_feature_builder_returns_dataframe():
    import pandas as pd
    event = {
        "user_id": "u_test",
        "days_since_last_purchase": 30.0,
        "login_frequency": 5.0,
        "session_duration_avg": 8.0,
        "pages_per_session": 3.0,
    }
    df = build_online_features(event)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_feature_builder_columns():
    event = {"days_since_last_purchase": 10.0, "login_frequency": 8.0}
    df = build_online_features(event)
    assert "recency_days" in df.columns
    assert "frequency" in df.columns


def test_feature_builder_defaults():
    df = build_online_features({})
    assert df["recency_days"].iloc[0] == 0.0
    assert df["engagement_trend"].iloc[0] == 0.0


# ---- recommendations ----

@pytest.mark.parametrize("score,expected_seg", [
    (0.1, "engaged"),
    (0.5, "occasional"),
    (0.8, "at_risk"),
])
def test_score_to_segment(score, expected_seg):
    assert score_to_segment(score) == expected_seg


@pytest.mark.parametrize("score,expected_risk", [
    (0.2, "baixo"),
    (0.55, "medio"),
    (0.75, "alto"),
])
def test_score_to_risk(score, expected_risk):
    assert score_to_risk_level(score) == expected_risk


def test_recommend_priority_call():
    assert recommend_action("at_risk", 0.90) == "priority_retention_call"
