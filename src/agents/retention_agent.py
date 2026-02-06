# src/agents/retention_agent.py
from .action_policy import action_for_score

class RetentionAgent:
    """
    Agente que decide ações baseado no churn_score
    """
    def __init__(self, model):
        self.model = model

    def decide_action(self, user_features):
        # user_features deve ser array-like ou Series
        score = self.model.predict_proba([user_features])[0][1]
        action = action_for_score(score)
        return score, action
