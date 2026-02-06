from fastapi import FastAPI
import pandas as pd
from src.models.churn_model import train_models
from src.agents.retention_agent import decide_action

app = FastAPI()

# Carregar modelo pré-treinado (exemplo simples)
df = pd.read_csv("../data/processed/features.csv")
models_dict, _ = train_models(df)
rf_model = models_dict["RandomForest"]

@app.post("/predict")
def predict_churn(user_data: dict):
    X_input = pd.DataFrame([user_data])
    score = rf_model.predict_proba(X_input)[:,1][0]

    # segmentação simples
    if score < 0.3:
        segment = "engaged"
    elif score < 0.6:
        segment = "neutral"
    else:
        segment = "at_risk"

    action = decide_action(score, segment)
    return {"churn_score": score, "segment": segment, "action": action}
