# src/evaluation/langfuse_tracking.py

def log_decision(state: dict):
    print({
        "churn_score": state["churn_score"],
        "analysis": state.get("analysis"),
        "strategy": state.get("strategy"),
        "audit": state.get("audit")
    })
