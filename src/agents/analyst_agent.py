class AnalystAgent:
    """
    Agente analista:
    interpreta score, features e contexto RAG.
    Não decide ação.
    """

    def run(
        self,
        churn_score: float,
        features: dict,
        rag_context: str
    ) -> dict:

        risk_level = (
            "alto" if churn_score > 0.7
            else "medio" if churn_score > 0.3
            else "baixo"
        )

        return {
            "risk_level": risk_level,
            "churn_score": churn_score,
            "key_features": features,
            "rag_context": rag_context,
            "summary": (
                f"Risco {risk_level}. "
                f"Score={churn_score:.2f}. "
                f"Principais sinais: {list(features.keys())}."
            )
        }
