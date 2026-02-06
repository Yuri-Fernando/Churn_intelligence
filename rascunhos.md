# ============================================================
# 📓 run_pipeline_fase3_cognitive_fast.ipynb
# Fase 3 – ML → filtro → agentes → RAG → decisão auditada
# (SEM MLflow, SEM gargalo, UMA CÉLULA)
# ============================================================

import sys, os, warnings
sys.path.append(os.path.abspath(".."))
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import numpy as np

from src.agents.retention_agent import RetentionAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.auditor_agent import AuditorAgent

from src.llm.rag import ChurnRAG
from src.llm.retriever import SimpleRetriever
from src.llm.generator import LLMGenerator

from src.privacy.bias_analysis import analyze_bias


# -----------------------------
# 1️⃣ Dados + modelo (igual Fase 2)
# -----------------------------
df = pd.read_csv("../data/processed/features.csv")
rf_model = joblib.load("../models/rf_model.pkl")

X_input = df[rf_model.feature_names_in_]
retention_agent = RetentionAgent(rf_model)


# -----------------------------
# 2️⃣ Inferência ML (BATCH SIMPLES)
# -----------------------------
scores = []
base_actions = []

for _, row in X_input.iterrows():
    score, action = retention_agent.decide_action(row)
    scores.append(float(score))
    base_actions.append(action)

df["churn_score"] = scores
df["base_action"] = base_actions


# -----------------------------
# 3️⃣ Filtro cognitivo (chave da performance)
# -----------------------------
cognitive_mask = (df["churn_score"] >= 0.4) & (df["churn_score"] <= 0.7)
df_cog = df[cognitive_mask]


# -----------------------------
# 4️⃣ Inicializa RAG + agentes (uma vez)
# -----------------------------
rag = ChurnRAG(
    retriever=SimpleRetriever(),
    generator=LLMGenerator(llm_client=None)  # mock / stub
)

analyst_agent = AnalystAgent()
strategy_agent = StrategyAgent()
auditor_agent = AuditorAgent()


# -----------------------------
# 5️⃣ Execução cognitiva (SÓ onde importa)
# -----------------------------
final_actions = df["base_action"].copy()
audit_status = ["skipped"] * len(df)
rag_explanations = [None] * len(df)

for idx, row in df_cog.iterrows():

    features = {
        "recency_days": row.get("recency_days"),
        "frequency": row.get("frequency"),
        "avg_session_duration": row.get("avg_session_duration"),
        "intensity": row.get("intensity"),
    }

    rag_context = rag.run(
        user_id=str(idx),
        churn_score=row["churn_score"],
        features=features
    )

    analysis = analyst_agent.run(
        churn_score=row["churn_score"],
        features=features,
        rag_context=rag_context
    )

    decision = strategy_agent.run(
        analysis=analysis,
        previous_action=row["base_action"]
    )

    state = {
        "churn_score": row["churn_score"],
        "strategy": {"action": decision["action"]},
        "analysis": analysis,
        "rag_context": rag_context
    }

    audit = auditor_agent.run(state)

    final_actions.loc[idx] = decision["action"]
    audit_status[idx] = audit["audit"]["status"]
    rag_explanations[idx] = rag_context


# -----------------------------
# 6️⃣ Consolidação
# -----------------------------
df["final_action"] = final_actions
df["audit_status"] = audit_status
df["rag_explanation"] = rag_explanations

df["segment"] = df["churn_score"].apply(
    lambda x: "engaged" if x < 0.3 else "neutral" if x < 0.7 else "at_risk"
)


# -----------------------------
# 7️⃣ Visão executiva
# -----------------------------
summary = (
    df.groupby(["segment", "final_action"])
      .size()
      .reset_index(name="num_users")
)

display(summary)


# -----------------------------
# 8️⃣ Fairness / Bias
# -----------------------------
bias_report = analyze_bias(df)
print("Relatório de bias:", bias_report)


# -----------------------------
# 9️⃣ Demo explicável
# -----------------------------
sample = df.sample(1).iloc[0]

print("\n--- DEMO EXPLICÁVEL ---")
print("Churn score:", sample["churn_score"])
print("Ação base:", sample["base_action"])
print("Ação final:", sample["final_action"])
print("Auditoria:", sample["audit_status"])
print("\nRAG explanation:\n", sample["rag_explanation"])

print("\n✅ Fase 3 executada com sucesso (rápida, estável, sem gargalo).")


___________________________________________________________________________________

# ============================================================
# 📓 run_pipeline_fase3_churn_rag_agents_mlflow.ipynb
# Fase 3 – Pipeline Cognitivo Completo (UMA CÉLULA)
# ML → Segmentação → Agentes → RAG → Decisão auditada → MLflow
# ============================================================

import sys, os, warnings
sys.path.append(os.path.abspath(".."))
warnings.filterwarnings("ignore")

import pandas as pd
import joblib
import numpy as np
import mlflow

# -----------------------------
# Imports internos
# -----------------------------
from src.agents.retention_agent import RetentionAgent
from src.agents.analyst_agent import AnalystAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.auditor_agent import AuditorAgent

from src.llm.rag import ChurnRAG
from src.llm.retriever import SimpleRetriever
from src.llm.generator import LLMGenerator

from src.privacy.bias_analysis import analyze_bias

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("file:../mlruns")
mlflow.set_experiment("churn_cognitive_pipeline")
if mlflow.active_run():
    mlflow.end_run()

# -----------------------------
# 1️⃣ Dados + modelo
# -----------------------------
df = pd.read_csv("../data/processed/features.csv")
rf_model = joblib.load("../models/rf_model.pkl")
X_input = df[rf_model.feature_names_in_]

retention_agent = RetentionAgent(rf_model)

# -----------------------------
# 2️⃣ RAG
# -----------------------------
retriever = SimpleRetriever()
generator = LLMGenerator(llm_client=None)  # stub / mock
rag = ChurnRAG(retriever=retriever, generator=generator)

# -----------------------------
# 3️⃣ Inicializa agentes especialistas
# -----------------------------
analyst_agent = AnalystAgent()
strategy_agent = StrategyAgent()
auditor_agent = AuditorAgent()

# -----------------------------
# 4️⃣ Execução cognitiva
# -----------------------------
scores = []
base_actions = []
final_actions = []
rag_explanations = []
audit_status = []

with mlflow.start_run(run_name="fase3_churn_rag_agents"):

    for user_id, row in X_input.iterrows():
        # Modelo de churn
        churn_score, base_action = retention_agent.decide_action(row)

        features = {
            "recency_days": row.get("recency_days"),
            "frequency": row.get("frequency"),
            "avg_session_duration": row.get("avg_session_duration"),
            "intensity": row.get("intensity"),
        }

        # RAG (interpretativo)
        rag_context = rag.run(
            user_id=str(user_id),
            churn_score=float(churn_score),
            features=features
        )

        # Agente Analista
        analysis = analyst_agent.run(
            churn_score=float(churn_score),
            features=features,
            rag_context=rag_context
        )

        # Agente Estratégia
        decision = strategy_agent.run(
            analysis=analysis,
            previous_action=base_action
        )

        # Agente Auditor – tudo dentro de state
        state = {
            "churn_score": churn_score,
            "strategy": {"action": decision["action"]},
            "analysis": analysis,
            "rag_context": rag_context
        }
        audit = auditor_agent.run(state)

        # Salva resultados
        scores.append(float(churn_score))
        base_actions.append(base_action)
        final_actions.append(decision["action"])
        rag_explanations.append(rag_context)
        audit_status.append(audit["audit"]["status"])

    # -----------------------------
    # 5️⃣ MLflow – métricas agregadas
    # -----------------------------
    mlflow.log_metric("churn_score_mean", np.mean(scores))
    mlflow.log_metric("churn_score_std", np.std(scores))
    mlflow.log_metric("churn_score_min", np.min(scores))
    mlflow.log_metric("churn_score_max", np.max(scores))

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("pipeline_phase", "fase_3")
    mlflow.log_param("rag_enabled", True)
    mlflow.log_param("agents", "analyst|strategy|auditor")

# -----------------------------
# 6️⃣ Pós-processamento
# -----------------------------
df["churn_score"] = scores
df["base_action"] = base_actions
df["final_action"] = final_actions
df["rag_explanation"] = rag_explanations
df["audit_status"] = audit_status

df["segment"] = df["churn_score"].apply(
    lambda x: "engaged" if x < 0.3 else "neutral" if x < 0.7 else "at_risk"
)

# -----------------------------
# 7️⃣ Visão executiva
# -----------------------------
summary = (
    df.groupby(["segment", "final_action"])
      .size()
      .reset_index(name="num_users")
)
display(summary)

# -----------------------------
# 8️⃣ Fairness / Bias
# -----------------------------
bias_report = analyze_bias(df)
print("Relatório de bias:", bias_report)

# -----------------------------
# 9️⃣ Demo explicável
# -----------------------------
sample = df.sample(1).iloc[0]

print("\n--- DEMO EXPLICÁVEL ---")
print("Churn score:", sample["churn_score"])
print("Ação base:", sample["base_action"])
print("Ação final:", sample["final_action"])
print("Auditoria:", sample["audit_status"])
print("\nRAG explanation:\n", sample["rag_explanation"])

print("\n✅ Pipeline Fase 3 finalizado com sucesso.")
