"""
Logica de recomendacao de acoes baseada em segmento e score de churn.
"""

ACTION_MAP = {
    "at_risk": "offer_discount",
    "inactive": "reengagement_email",
    "occasional": "engagement_campaign",
    "engaged": "recommend_new_product",
    "neutral": "engagement_campaign",
}


def recommend_action(segment: str, churn_score: float) -> str:
    """Retorna acao recomendada com base no segmento e score."""
    if churn_score >= 0.85:
        return "priority_retention_call"
    return ACTION_MAP.get(segment, "no_action")


def score_to_risk_level(score: float) -> str:
    if score >= 0.7:
        return "alto"
    elif score >= 0.4:
        return "medio"
    return "baixo"


def score_to_segment(score: float) -> str:
    if score >= 0.7:
        return "at_risk"
    elif score >= 0.4:
        return "occasional"
    return "engaged"
