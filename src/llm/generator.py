"""
LLM Generator com fallback gracioso.

Se OPENAI_API_KEY estiver configurada, usa GPT-4o-mini para gerar
explicacoes narrativas reais. Caso contrario, retorna template formatado.
"""

import os


PROMPT_TEMPLATE = """Voce e um especialista em retencao de clientes.

Score de churn: {score:.2f}
Contexto comportamental: {context}
Features principais: {features}

Explique em 2-3 frases:
1. Por que este usuario esta em risco
2. Qual acao de retencao faz mais sentido

Seja direto e baseie-se apenas nos dados fornecidos."""


class LLMGenerator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._openai_available = self._check_openai()

    def _check_openai(self) -> bool:
        try:
            import openai  # noqa: F401
            return bool(os.getenv("OPENAI_API_KEY"))
        except ImportError:
            return False

    def generate(
        self,
        user_id: str,
        context: str,
        churn_score: float,
        features: dict,
    ) -> str:
        prompt = PROMPT_TEMPLATE.format(
            score=churn_score,
            context=context,
            features=features,
        )

        if self._openai_available:
            return self._call_openai(prompt, user_id)

        return self._fallback(user_id, churn_score, context)

    def _call_openai(self, prompt: str, user_id: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return self._fallback(user_id, 0.0, f"[openai error: {exc}]")

    def _fallback(self, user_id: str, churn_score: float, context: str) -> str:
        risk = "alto" if churn_score >= 0.7 else "medio" if churn_score >= 0.4 else "baixo"
        return (
            f"[{user_id}] Risco {risk} de churn (score={churn_score:.2f}). "
            f"{context} "
            f"(LLM nao configurado — defina OPENAI_API_KEY para explicacoes narrativas reais.)"
        )
