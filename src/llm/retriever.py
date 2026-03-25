"""
Retriever com busca vetorial via sentence-transformers.

Se sentence-transformers estiver disponivel, usa embeddings reais para
buscar o contexto mais relevante de uma base de conhecimento.
Fallback para template simples se nao estiver instalado.
"""

from __future__ import annotations
import os
from typing import List


# Base de conhecimento: regras de negocio + playbooks de retencao
KNOWLEDGE_BASE = [
    "Usuarios com mais de 60 dias sem interacao tem 3x mais chance de churn permanente.",
    "Desconto direto e eficaz para usuarios de alto valor (LTV > media) com score > 0.7.",
    "Campanha de engajamento por email tem melhor ROI para usuarios com score entre 0.4 e 0.7.",
    "Usuarios com alta frequencia mas baixa intensidade respondem bem a recomendacoes de produto.",
    "Reativacao por push notification e mais eficaz para usuarios que abandonaram carrinho recentemente.",
    "Usuarios com queda de frequencia nas ultimas 3 semanas sao os mais receptivos a ofertas limitadas.",
    "Ligacao de retencao direta e recomendada apenas para usuarios com LTV no top 10%.",
    "Usuarios recentemente adquiridos (< 30 dias) raramente respondem a campanhas de desconto.",
]


def build_context(churn_score: float, features: dict) -> str:
    """Fallback: contexto por template."""
    recency  = features.get("recency_days", "?")
    freq     = features.get("frequency", "?")
    duration = features.get("avg_session_duration", "?")
    return (
        f"Score de churn: {churn_score:.2f}. "
        f"Recencia: {recency} dias. "
        f"Frequencia: {freq} logins. "
        f"Sessao media: {duration} min."
    )


class SimpleRetriever:
    """Retriever com embeddings reais (fallback para template)."""

    def __init__(self, knowledge_base: List[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        self._kb = knowledge_base or KNOWLEDGE_BASE
        self._model = None
        self._embeddings = None
        self._model_name = model_name
        self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            self._model = SentenceTransformer(self._model_name)
            self._embeddings = self._model.encode(self._kb, convert_to_numpy=True)
        except Exception:
            pass  # fallback gracioso

    def retrieve(self, churn_score: float, features: dict, top_k: int = 2) -> str:
        """Retorna contexto relevante da base de conhecimento."""
        if self._model is None:
            return build_context(churn_score, features)

        try:
            import numpy as np
            query = build_context(churn_score, features)
            q_emb = self._model.encode([query], convert_to_numpy=True)
            # cosine similarity
            norms = (
                np.linalg.norm(self._embeddings, axis=1) *
                np.linalg.norm(q_emb)
            )
            sims = (self._embeddings @ q_emb.T).flatten() / (norms + 1e-9)
            top_idx = sims.argsort()[-top_k:][::-1]
            retrieved = " | ".join(self._kb[i] for i in top_idx)
            return f"{build_context(churn_score, features)} Contexto relevante: {retrieved}"
        except Exception:
            return build_context(churn_score, features)
