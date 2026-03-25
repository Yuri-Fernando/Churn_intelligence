import pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.analyst_agent import AnalystAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.auditor_agent import AuditorAgent
from src.agents.action_policy import action_for_score
from src.api.recommendations import recommend_action, score_to_risk_level, score_to_segment


SAMPLE_FEATURES = {"recency_days": 45.0, "frequency": 3.0, "avg_session_duration": 5.0}


# ---- action_policy ----

def test_action_for_low_score():
    assert action_for_score(0.1) == "recommend_new_product"

def test_action_for_medium_score():
    assert action_for_score(0.5) == "no_action"

def test_action_for_high_score():
    assert action_for_score(0.9) == "offer_discount"


# ---- recommendations ----

def test_score_to_risk_high():
    assert score_to_risk_level(0.8) == "alto"

def test_score_to_risk_medium():
    assert score_to_risk_level(0.5) == "medio"

def test_score_to_risk_low():
    assert score_to_risk_level(0.2) == "baixo"

def test_recommend_action_at_risk():
    assert recommend_action("at_risk", 0.75) == "offer_discount"

def test_recommend_action_priority():
    assert recommend_action("at_risk", 0.90) == "priority_retention_call"

def test_recommend_action_engaged():
    assert recommend_action("engaged", 0.2) == "recommend_new_product"


# ---- AnalystAgent ----

def test_analyst_high_risk():
    agent = AnalystAgent()
    result = agent.run(churn_score=0.85, features=SAMPLE_FEATURES, rag_context="")
    assert result["risk_level"] == "alto"

def test_analyst_medium_risk():
    agent = AnalystAgent()
    result = agent.run(churn_score=0.5, features=SAMPLE_FEATURES, rag_context="")
    assert result["risk_level"] == "medio"

def test_analyst_low_risk():
    agent = AnalystAgent()
    result = agent.run(churn_score=0.1, features=SAMPLE_FEATURES, rag_context="")
    assert result["risk_level"] == "baixo"

def test_analyst_returns_summary():
    agent = AnalystAgent()
    result = agent.run(churn_score=0.5, features=SAMPLE_FEATURES, rag_context="ctx")
    assert "summary" in result
    assert isinstance(result["summary"], str)


# ---- StrategyAgent ----

def test_strategy_high_risk_action():
    agent = StrategyAgent()
    analysis = {"risk_level": "alto", "churn_score": 0.8}
    result = agent.run(analysis=analysis, previous_action="no_action")
    assert result["action"] == "offer_discount"

def test_strategy_medium_risk_action():
    agent = StrategyAgent()
    analysis = {"risk_level": "medio", "churn_score": 0.5}
    result = agent.run(analysis=analysis, previous_action="no_action")
    assert result["action"] == "engagement_campaign"

def test_strategy_low_risk_uses_previous():
    agent = StrategyAgent()
    analysis = {"risk_level": "baixo", "churn_score": 0.1}
    result = agent.run(analysis=analysis, previous_action="recommend_new_product")
    assert result["action"] == "recommend_new_product"


# ---- AuditorAgent ----

def test_auditor_ok_decision():
    agent = AuditorAgent()
    state = {
        "churn_score": 0.8,
        "strategy": {"action": "offer_discount"},
    }
    result = agent.run(state)
    assert result["audit"]["status"] == "ok"

def test_auditor_flags_inconsistent():
    agent = AuditorAgent()
    state = {
        "churn_score": 0.9,
        "strategy": {"action": "send_engagement_reminder"},
    }
    result = agent.run(state)
    assert result["audit"]["status"] == "review"
    assert "inconsistent_decision" in result["audit"]["flags"]
