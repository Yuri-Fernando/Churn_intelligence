# Funcoes de criacao de features online
"""
Feature builder para predicao em tempo quase real.

Recebe um dicionario com campos brutos de um evento de usuario
e retorna um DataFrame com as features no formato esperado pelo modelo.
"""

import pandas as pd


FEATURE_DEFAULTS = {
    "recency_days": 0.0,
    "frequency": 0.0,
    "avg_session_duration": 0.0,
    "intensity": 0.0,
    "engagement_trend": 0.0,
}


def build_online_features(raw_event: dict) -> pd.DataFrame:
    """
    Transforma evento raw em features prontas para o modelo.

    Mapeamentos:
      days_since_last_purchase -> recency_days
      login_frequency          -> frequency
      session_duration_avg     -> avg_session_duration
      pages_per_session        -> intensity
    """
    features = FEATURE_DEFAULTS.copy()

    if "days_since_last_purchase" in raw_event:
        features["recency_days"] = float(raw_event["days_since_last_purchase"])
    if "login_frequency" in raw_event:
        features["frequency"] = float(raw_event["login_frequency"])
    if "session_duration_avg" in raw_event:
        features["avg_session_duration"] = float(raw_event["session_duration_avg"])
    if "pages_per_session" in raw_event:
        features["intensity"] = float(raw_event["pages_per_session"])

    # Campos ja normalizados passam direto
    for key in FEATURE_DEFAULTS:
        if key in raw_event and key not in features:
            features[key] = float(raw_event[key])

    return pd.DataFrame([features])
