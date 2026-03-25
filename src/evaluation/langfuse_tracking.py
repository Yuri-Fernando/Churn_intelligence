"""
Langfuse tracking com fallback gracioso.

Se LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY estiverem configuradas,
loga decisoes no Langfuse. Caso contrario, imprime localmente.

Configuracao:
  export LANGFUSE_PUBLIC_KEY="pk-..."
  export LANGFUSE_SECRET_KEY="sk-..."
  export LANGFUSE_HOST="https://cloud.langfuse.com"  # opcional
"""

import os
import json
from datetime import datetime


def _get_langfuse():
    """Retorna cliente Langfuse ou None se nao configurado."""
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    if not (pk and sk):
        return None
    try:
        from langfuse import Langfuse
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return Langfuse(public_key=pk, secret_key=sk, host=host)
    except ImportError:
        return None


def log_decision(state: dict) -> None:
    """
    Loga decisao cognitiva completa.

    state deve conter: churn_score, analysis, strategy, audit
    """
    lf = _get_langfuse()
    user_id = state.get("user_id", "unknown")
    score   = state.get("churn_score", 0.0)
    action  = state.get("strategy", {}).get("action", "")
    risk    = state.get("analysis", {}).get("risk_level", "")
    audit   = state.get("audit", {}).get("status", "ok")

    if lf:
        trace = lf.trace(
            name="churn_decision",
            user_id=str(user_id),
            metadata={
                "churn_score": score,
                "risk_level":  risk,
                "action":      action,
                "audit_status": audit,
            },
        )
        trace.generation(
            name="strategy_decision",
            input={"analysis": state.get("analysis")},
            output={"strategy": state.get("strategy")},
            metadata={"score": score},
        )
        lf.flush()
    else:
        _local_log(state)


def log_rag_explanation(user_id: str, prompt: str, context: str, response: str) -> None:
    """Loga chamada RAG/LLM com prompt, contexto e resposta."""
    lf = _get_langfuse()

    if lf:
        trace = lf.trace(name="rag_explanation", user_id=str(user_id))
        trace.generation(
            name="llm_generate",
            input={"prompt": prompt, "context": context},
            output={"response": response},
        )
        lf.flush()
    else:
        _local_log({
            "event": "rag_explanation",
            "user_id": user_id,
            "response_preview": response[:100],
        })


def _local_log(data: dict) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
    print(f"[langfuse:{ts}] {json.dumps(data, default=str, ensure_ascii=False)}")
