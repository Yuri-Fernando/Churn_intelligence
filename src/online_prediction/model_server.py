# API para predicao em tempo quase real
"""
Servidor de predicao online.

Simula o fluxo de um evento de usuario chegando em tempo quase real:
  1. Evento raw com features basicas do usuario
  2. Feature builder constroi o vetor de entrada
  3. Modelo carregado faz predicao
  4. Acao e retornada junto com o score

Uso (terminal):
  uvicorn src.online_prediction.model_server:app --host 0.0.0.0 --port 8001
"""

import os
import sys
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.online_prediction.feature_builder import build_online_features
from src.api.recommendations import recommend_action, score_to_risk_level, score_to_segment

app = FastAPI(title="Churn Online Prediction Server", version="1.0.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/rf_model.pkl")
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


class RawEvent(BaseModel):
    user_id: str
    days_since_last_purchase: float
    login_frequency: float
    session_duration_avg: float
    pages_per_session: float


class OnlinePrediction(BaseModel):
    user_id: str
    churn_score: float
    segment: str
    action: str
    risk_level: str


@app.get("/health")
def health():
    return {"status": "ok", "service": "online_prediction"}


@app.post("/score", response_model=OnlinePrediction)
def score_event(event: RawEvent):
    try:
        features = build_online_features(event.dict())
        model = _get_model()

        if hasattr(model, "feature_names_in_"):
            features = features.reindex(columns=model.feature_names_in_, fill_value=0)

        score = float(model.predict_proba(features)[0][1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    segment = score_to_segment(score)
    return OnlinePrediction(
        user_id=event.user_id,
        churn_score=round(score, 4),
        segment=segment,
        action=recommend_action(segment, score),
        risk_level=score_to_risk_level(score),
    )
