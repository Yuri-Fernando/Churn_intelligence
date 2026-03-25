from pydantic import BaseModel, Field
from typing import Optional


class UserEventInput(BaseModel):
    user_id: str = Field(..., description="ID pseudonimizado do usuario")
    recency_days: float = Field(..., ge=0, description="Dias desde a ultima interacao")
    frequency: float = Field(..., ge=0, description="Frequencia de logins")
    avg_session_duration: float = Field(..., ge=0, description="Duracao media de sessao (minutos)")
    intensity: float = Field(..., ge=0, description="Paginas por sessao")
    engagement_trend: float = Field(0.0, description="Tendencia de engajamento (-1 a 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "u_abc123",
                "recency_days": 15.0,
                "frequency": 8.0,
                "avg_session_duration": 12.5,
                "intensity": 4.2,
                "engagement_trend": -0.1,
            }
        }


class PredictionOutput(BaseModel):
    user_id: str
    churn_score: float = Field(..., ge=0.0, le=1.0)
    segment: str
    action: str
    risk_level: str
    explanation: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
