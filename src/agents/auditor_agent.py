# src/agents/auditor_agent.py

class AuditorAgent:
    def run(self, state: dict) -> dict:
        """
        state: dict que deve conter pelo menos:
            - 'churn_score': float
            - 'strategy': dict com chave 'action'
        Pode opcionalmente conter:
            - 'analysis': qualquer info extra
            - 'rag_context': contexto adicional
        """

        churn_score = state.get("churn_score", 0)
        action = state.get("strategy", {}).get("action", None)
        analysis = state.get("analysis")      # opcional
        rag_context = state.get("rag_context")  # opcional

        flags = []

        # Exemplo de regra de auditoria
        if churn_score > 0.8 and action == "send_engagement_reminder":
            flags.append("inconsistent_decision")

        # Aqui você poderia adicionar regras usando analysis ou rag_context
        # Exemplo:
        # if analysis and analysis.get("some_metric") > 0.9:
        #     flags.append("check_metric")

        state["audit"] = {
            "flags": flags,
            "status": "ok" if not flags else "review"
        }

        return state
