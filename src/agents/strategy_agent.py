class StrategyAgent:
    """
    Agente de decisão estratégica.
    Recebe análise cognitiva + ação base do modelo.
    """

    def run(
        self,
        analysis: dict,
        previous_action: str
    ) -> dict:

        risk = analysis.get("risk_level")
        churn_score = analysis.get("churn_score")

        # Estratégia simples e auditável
        if risk == "alto":
            action = "offer_discount"
        elif risk == "medio":
            action = "engagement_campaign"
        else:
            action = previous_action

        return {
            "action": action,
            "reasoning": (
                f"Risco={risk}, score={churn_score:.2f}. "
                f"Ação base='{previous_action}'."
            ),
            "analysis_used": True
        }
