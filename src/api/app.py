"""
FastAPI – Churn Intelligence API

Endpoints:
  GET  /health      → status do servico
  POST /predict     → score de churn + segmento + acao recomendada
  POST /batch       → predicao em lote (lista de usuarios)
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from typing import List

# Ajusta path para importacoes relativas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.api.schemas import UserEventInput, PredictionOutput, HealthResponse
from src.api.recommendations import recommend_action, score_to_risk_level, score_to_segment

app = FastAPI(
    title="Churn Intelligence API",
    description="Predição de churn em tempo quase real com agentes de retenção",
    version="1.0.0",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/rf_model.pkl")
_model = None


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}. Execute o pipeline de treinamento primeiro.")
        _model = joblib.load(MODEL_PATH)
    return _model


FEATURE_ORDER = [
    "recency_days", "frequency", "avg_session_duration",
    "intensity", "engagement_trend",
]


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        _load_model()
        loaded = True
    except FileNotFoundError:
        loaded = False
    return {"status": "ok", "model_loaded": loaded}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: UserEventInput):
    model = _load_model()

    features = pd.DataFrame([{
        "recency_days": payload.recency_days,
        "frequency": payload.frequency,
        "avg_session_duration": payload.avg_session_duration,
        "intensity": payload.intensity,
        "engagement_trend": payload.engagement_trend,
    }])

    # Alinha colunas com o modelo treinado
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in features.columns:
                features[col] = 0
        try:
            features = features[expected]
        except KeyError:
            features = features.reindex(columns=expected, fill_value=0)

    try:
        score = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Erro na predição: {exc}")

    segment = score_to_segment(score)
    risk = score_to_risk_level(score)
    action = recommend_action(segment, score)
    explanation = (
        f"Score de churn: {score:.2f}. Risco {risk}. "
        f"Recência: {payload.recency_days:.0f}d, "
        f"Frequência: {payload.frequency:.0f} logins. "
        f"Ação sugerida: {action}."
    )

    return PredictionOutput(
        user_id=payload.user_id,
        churn_score=round(score, 4),
        segment=segment,
        action=action,
        risk_level=risk,
        explanation=explanation,
    )


@app.post("/batch", response_model=List[PredictionOutput])
def predict_batch(payloads: List[UserEventInput]):
    return [predict(p) for p in payloads]
