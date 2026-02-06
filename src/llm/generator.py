class LLMGenerator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def generate(
        self,
        user_id: str,
        context: str,
        churn_score: float,
        features: dict
    ) -> str:
        return (
            f"[User {user_id}] "
            f"Churn score={churn_score:.2f}. "
            f"Contexto: {context}"
        )
