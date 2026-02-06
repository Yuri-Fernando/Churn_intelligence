class ChurnRAG:
    """
    Camada RAG interpretativa para churn.
    Apenas contextualiza e explica.
    """

    def __init__(self, retriever, generator):
        self._retriever = retriever
        self._generator = generator

    def run(self, user_id: str, churn_score: float, features: dict) -> str:
        context = self._retriever.retrieve(
            churn_score=churn_score,
            features=features
        )

        return self._generator.generate(
            user_id=user_id,
            context=context,
            churn_score=churn_score,
            features=features
        )
