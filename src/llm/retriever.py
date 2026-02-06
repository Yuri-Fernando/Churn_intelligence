def build_context(churn_score: float, features: dict) -> str:
    return (
        f"Churn score: {churn_score:.2f}. "
        f"Features relevantes: {features}."
    )

class SimpleRetriever:
    def retrieve(self, churn_score: float, features: dict) -> str:
        return build_context(churn_score, features)
