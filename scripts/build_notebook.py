"""Script para gerar main.ipynb reorganizado com todas as fases."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT  = os.path.join(ROOT, "notebooks", "main.ipynb")


def md(source):
    if isinstance(source, str):
        source = [source]
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def co(source):
    if isinstance(source, str):
        source = [source]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
cells.append(md([
    "# Churn Intelligence Pipeline\n",
    "\n",
    "Pipeline completo: engenharia de features, modelos de churn, segmentacao dinamica,\n",
    "agentes cognitivos, RAG, API online e analise de fairness.\n",
    "\n",
    "| Fase | Descricao |\n",
    "|------|-----------|\n",
    "| 1 | Engenharia de features + treinamento de modelos |\n",
    "| 2 | Segmentacao dinamica (KMeans + regras) + agentes + MLflow |\n",
    "| 3 | Camada cognitiva: RAG + agentes especializados + auditoria |\n",
    "| 4 | API de predicao online (simulacao) |\n",
    "| 5 | Privacidade, anonimizacao e analise de fairness |\n",
]))

# ------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------
cells.append(md("## Setup\n"))
cells.append(co([
    "import sys, os, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Sobe da pasta notebooks/ para a raiz do projeto\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "import mlflow\n",
    "\n",
    "print('Setup concluido')\n",
]))

# ------------------------------------------------------------------
# FASE 1
# ------------------------------------------------------------------
cells.append(md([
    "---\n",
    "## Fase 1 -- Engenharia de Features + Modelos de Churn\n",
    "\n",
    "**Objetivo:** pipeline basico funcional -- dados processados, tres modelos treinados e metricas.\n",
    "\n",
    "Features construidas a partir do dataset bruto:\n",
    "- `recency_days` -- dias desde a ultima interacao\n",
    "- `frequency` -- frequencia de logins\n",
    "- `avg_session_duration` -- duracao media de sessao (min)\n",
    "- `intensity` -- paginas por sessao\n",
    "- `engagement_trend` -- placeholder de tendencia de engajamento\n",
]))

cells.append(co([
    "from src.features.build_features import build_features\n",
    "\n",
    "df = build_features(\n",
    "    input_csv='../data/raw/ecommerce_customer_churn_dataset.csv',\n",
    "    output_csv='../data/processed/features.csv',\n",
    ")\n",
    "\n",
    "print(f'Dataset: {df.shape[0]:,} usuarios | {df.shape[1]} colunas')\n",
    "print(f'Churn rate global: {df[\"churn\"].mean():.1%}')\n",
    "df.head(3)\n",
]))

cells.append(co([
    "from src.models.churn_model import train_models\n",
    "from src.evaluation.metrics import print_metrics\n",
    "\n",
    "models_dict, results_dict = train_models(df)\n",
    "\n",
    "print('\\n=== Metricas por modelo ===')\n",
    "print_metrics(results_dict)\n",
]))

cells.append(co([
    "# Salvar Random Forest para uso nas fases seguintes\n",
    "os.makedirs('../models', exist_ok=True)\n",
    "joblib.dump(models_dict['RandomForest'], '../models/rf_model.pkl')\n",
    "print('Modelo salvo em ../models/rf_model.pkl')\n",
]))

cells.append(co([
    "# SHAP -- importancia de features (requer pacote shap)\n",
    "try:\n",
    "    from src.evaluation.shap_analysis import explain_model\n",
    "    from src.models.model_utils import split_features_target\n",
    "    X_train, X_test, y_train, y_test = split_features_target(df)\n",
    "    explain_model(models_dict['RandomForest'], X_train)\n",
    "except Exception as e:\n",
    "    print(f'SHAP nao disponivel: {e}')\n",
]))

# ------------------------------------------------------------------
# FASE 2
# ------------------------------------------------------------------
cells.append(md([
    "---\n",
    "## Fase 2 -- Segmentacao Dinamica + Agentes + MLflow\n",
    "\n",
    "**Objetivo:** transformar scores de churn em segmentos acionaveis e rastrear experimentos.\n",
    "\n",
    "- **KMeans**: segmentacao nao-supervisionada por comportamento (4 clusters)\n",
    "- **Regras**: limiar de recencia + frequencia como baseline\n",
    "- **RetentionAgent**: associa acao a cada usuario pelo score\n",
    "- **MLflow**: rastreia metricas e parametros do experimento\n",
]))

cells.append(co([
    "from src.models.clustering_model import cluster_users, cluster_summary\n",
    "\n",
    "df_clustered = cluster_users(df, n_clusters=4)\n",
    "\n",
    "print('=== Segmentacao KMeans ===')\n",
    "print(df_clustered['cluster_label'].value_counts().to_string())\n",
    "print()\n",
    "print('=== Media de features por cluster ===')\n",
    "display(cluster_summary(df_clustered))\n",
]))

cells.append(co([
    "from src.segmentation.user_segmentation import segment_users\n",
    "from src.personalization.actions import generate_action\n",
    "\n",
    "df_seg = segment_users(df)\n",
    "df_seg['action'] = df_seg['segment'].apply(generate_action)\n",
    "\n",
    "summary = (\n",
    "    df_seg.groupby(['segment', 'action'])\n",
    "    .size()\n",
    "    .reset_index(name='usuarios')\n",
    "    .sort_values('usuarios', ascending=False)\n",
    ")\n",
    "print('=== Segmentacao por regras ===')\n",
    "display(summary)\n",
]))

cells.append(co([
    "from src.agents.retention_agent import RetentionAgent\n",
    "from src.models.model_utils import split_features_target\n",
    "\n",
    "rf_model = joblib.load('../models/rf_model.pkl')\n",
    "agent = RetentionAgent(rf_model)\n",
    "\n",
    "X_train, X_test, y_train, y_test = split_features_target(df)\n",
    "churn_scores = rf_model.predict_proba(X_test)[:, 1]\n",
    "\n",
    "df_test = X_test.copy()\n",
    "df_test['churn_score'] = churn_scores\n",
    "df_test['user_id'] = [f'u_{i:05d}' for i in range(len(df_test))]\n",
    "\n",
    "print(f'Score medio: {churn_scores.mean():.3f}')\n",
    "print(f'Alto risco (score > 0.7): {(churn_scores > 0.7).sum():,} usuarios ({(churn_scores > 0.7).mean():.1%})')\n",
]))

cells.append(co([
    "# MLflow -- registrar metricas do experimento\n",
    "mlflow.set_tracking_uri('sqlite:///../notebooks/mlflow.db')\n",
    "mlflow.set_experiment('churn_pipeline_fase2')\n",
    "\n",
    "from src.models.model_utils import evaluate_model\n",
    "\n",
    "with mlflow.start_run(run_name='rf_segmentation'):\n",
    "    metrics = evaluate_model(rf_model, X_test, y_test)\n",
    "    mlflow.log_metrics({\n",
    "        'roc_auc':          metrics['roc_auc'],\n",
    "        'precision':        metrics['precision'],\n",
    "        'recall':           metrics['recall'],\n",
    "        'churn_score_mean': float(churn_scores.mean()),\n",
    "        'churn_score_std':  float(churn_scores.std()),\n",
    "        'high_risk_count':  float((churn_scores > 0.7).sum()),\n",
    "    })\n",
    "    mlflow.log_param('model', 'RandomForest')\n",
    "    mlflow.log_param('n_estimators', 100)\n",
    "\n",
    "print('Experimento registrado no MLflow.')\n",
    "print('Visualizar: mlflow ui --backend-store-uri sqlite:///../notebooks/mlflow.db')\n",
]))

# ------------------------------------------------------------------
# FASE 3
# ------------------------------------------------------------------
cells.append(md([
    "---\n",
    "## Fase 3 -- Camada Cognitiva: RAG + Agentes + Auditoria\n",
    "\n",
    "**Objetivo:** adicionar explicabilidade narrativa e governanca de decisao.\n",
    "\n",
    "**Filtro cognitivo:** score entre 0.4 e 0.7 (zona de incerteza) passa pelo pipeline completo.\n",
    "Scores extremos recebem decisao direta sem custo computacional extra.\n",
    "\n",
    "Pipeline:\n",
    "```\n",
    "Score -> [Filtro 0.4-0.7] -> AnalystAgent -> StrategyAgent -> RAG -> AuditorAgent\n",
    "```\n",
    "\n",
    "- **AnalystAgent**: interpreta score e features, define nivel de risco\n",
    "- **StrategyAgent**: cruza analise com regras de negocio, decide acao\n",
    "- **RAG**: gera explicacao narrativa contextualizada\n",
    "- **AuditorAgent**: valida consistencia da decisao e emite flags\n",
]))

cells.append(co([
    "from src.agents.analyst_agent import AnalystAgent\n",
    "from src.agents.strategy_agent import StrategyAgent\n",
    "from src.agents.auditor_agent import AuditorAgent\n",
    "from src.llm.retriever import SimpleRetriever\n",
    "from src.llm.generator import LLMGenerator\n",
    "from src.llm.rag import ChurnRAG\n",
    "\n",
    "analyst  = AnalystAgent()\n",
    "strategy = StrategyAgent()\n",
    "auditor  = AuditorAgent()\n",
    "rag      = ChurnRAG(SimpleRetriever(), LLMGenerator())\n",
    "\n",
    "COGNITIVE_MIN, COGNITIVE_MAX = 0.4, 0.7\n",
    "print('Agentes cognitivos carregados')\n",
]))

cells.append(co([
    "def run_cognitive_pipeline(user_id, churn_score, features):\n",
    "    \"\"\"Executa pipeline completo: Analyst -> Strategy -> RAG -> Auditor.\"\"\"\n",
    "    analysis    = analyst.run(churn_score=churn_score, features=features, rag_context='')\n",
    "    action_base = 'offer_discount' if churn_score >= 0.7 else 'engagement_campaign'\n",
    "    decision    = strategy.run(analysis=analysis, previous_action=action_base)\n",
    "    explanation = rag.run(user_id=user_id, churn_score=churn_score, features=features)\n",
    "    state = auditor.run({\n",
    "        'churn_score': churn_score,\n",
    "        'analysis':    analysis,\n",
    "        'strategy':    decision,\n",
    "        'rag_context': explanation,\n",
    "    })\n",
    "    return {\n",
    "        'user_id':      user_id,\n",
    "        'churn_score':  churn_score,\n",
    "        'risk_level':   analysis['risk_level'],\n",
    "        'action':       decision['action'],\n",
    "        'reasoning':    decision['reasoning'],\n",
    "        'explanation':  explanation,\n",
    "        'audit_status': state['audit']['status'],\n",
    "        'audit_flags':  state['audit']['flags'],\n",
    "    }\n",
    "\n",
    "\n",
    "mask   = (df_test['churn_score'] >= COGNITIVE_MIN) & (df_test['churn_score'] <= COGNITIVE_MAX)\n",
    "sample = df_test[mask].head(5)\n",
    "\n",
    "print(f'Usuarios no filtro cognitivo ({COGNITIVE_MIN}-{COGNITIVE_MAX}): {mask.sum():,} ({mask.mean():.1%})')\n",
    "print(f'Processando {len(sample)} usuarios de exemplo...\\n')\n",
    "\n",
    "results_cog = []\n",
    "for _, row in sample.iterrows():\n",
    "    feats = {k: round(float(v), 3)\n",
    "             for k, v in row.drop(['churn_score', 'user_id']).items()\n",
    "             if not pd.isna(v)}\n",
    "    r = run_cognitive_pipeline(str(row['user_id']), float(row['churn_score']), feats)\n",
    "    results_cog.append(r)\n",
    "    print(f\"{r['user_id']:10s} | score={r['churn_score']:.2f} | \"\n",
    "          f\"risco={r['risk_level']:5s} | acao={r['action']:22s} | audit={r['audit_status']}\")\n",
]))

cells.append(co([
    "decisions_df = pd.DataFrame(results_cog)[[\n",
    "    'user_id', 'churn_score', 'risk_level', 'action', 'reasoning', 'audit_status'\n",
    "]]\n",
    "decisions_df.to_csv('agent_decision_summary.csv', index=False)\n",
    "print('Resumo salvo em agent_decision_summary.csv')\n",
    "display(decisions_df)\n",
]))

# ------------------------------------------------------------------
# FASE 4
# ------------------------------------------------------------------
cells.append(md([
    "---\n",
    "## Fase 4 -- API de Predicao Online (Simulacao)\n",
    "\n",
    "**Objetivo:** demonstrar o fluxo de eventos em tempo quase real sem iniciar servidor HTTP.\n",
    "\n",
    "Em producao, iniciar os servidores:\n",
    "```bash\n",
    "# API principal (churn score + acao)\n",
    "uvicorn src.api.app:app --host 0.0.0.0 --port 8000\n",
    "\n",
    "# Servidor de predicao online (eventos raw)\n",
    "uvicorn src.online_prediction.model_server:app --host 0.0.0.0 --port 8001\n",
    "```\n",
    "\n",
    "Endpoints disponveis:\n",
    "- `GET  /health` -- status do servico\n",
    "- `POST /predict` -- predicao individual\n",
    "- `POST /batch` -- predicao em lote\n",
    "- `POST /score` -- predicao por evento raw (model_server)\n",
]))

cells.append(co([
    "from src.api.recommendations import recommend_action, score_to_risk_level, score_to_segment\n",
    "from src.online_prediction.feature_builder import build_online_features\n",
    "\n",
    "raw_events = [\n",
    "    {'user_id': 'u_alpha', 'days_since_last_purchase': 5,  'login_frequency': 20, 'session_duration_avg': 15, 'pages_per_session': 6},\n",
    "    {'user_id': 'u_beta',  'days_since_last_purchase': 45, 'login_frequency': 2,  'session_duration_avg': 4,  'pages_per_session': 1},\n",
    "    {'user_id': 'u_gamma', 'days_since_last_purchase': 90, 'login_frequency': 0,  'session_duration_avg': 0,  'pages_per_session': 0},\n",
    "    {'user_id': 'u_delta', 'days_since_last_purchase': 20, 'login_frequency': 8,  'session_duration_avg': 10, 'pages_per_session': 3},\n",
    "]\n",
    "\n",
    "print('=== Simulacao de Predicao Online ===\\n')\n",
    "print(f'{\"Usuario\":<12} {\"Score\":>6} {\"Risco\":<7} {\"Segmento\":<14} Acao')\n",
    "print('-' * 65)\n",
    "for ev in raw_events:\n",
    "    feat = build_online_features(ev)\n",
    "    if hasattr(rf_model, 'feature_names_in_'):\n",
    "        feat = feat.reindex(columns=rf_model.feature_names_in_, fill_value=0)\n",
    "    score   = float(rf_model.predict_proba(feat)[0][1])\n",
    "    segment = score_to_segment(score)\n",
    "    action  = recommend_action(segment, score)\n",
    "    risk    = score_to_risk_level(score)\n",
    "    print(f\"{ev['user_id']:<12} {score:>6.3f} {risk:<7} {segment:<14} {action}\")\n",
]))

cells.append(co([
    "import json\n",
    "\n",
    "payload = {\n",
    "    'user_id': 'u_demo_001',\n",
    "    'recency_days': 30.0,\n",
    "    'frequency': 5.0,\n",
    "    'avg_session_duration': 8.5,\n",
    "    'intensity': 2.8,\n",
    "    'engagement_trend': -0.3,\n",
    "}\n",
    "print('Payload para POST /predict:')\n",
    "print(json.dumps(payload, indent=2))\n",
    "print()\n",
    "print('Exemplo de resposta esperada:')\n",
    "print(json.dumps({\n",
    "    'user_id': 'u_demo_001',\n",
    "    'churn_score': 0.52,\n",
    "    'segment': 'occasional',\n",
    "    'action': 'engagement_campaign',\n",
    "    'risk_level': 'medio',\n",
    "    'explanation': 'Score de churn: 0.52. Risco medio. Recencia: 30d, Frequencia: 5 logins. Acao: engagement_campaign.',\n",
    "}, indent=2))\n",
]))

# ------------------------------------------------------------------
# FASE 5
# ------------------------------------------------------------------
cells.append(md([
    "---\n",
    "## Fase 5 -- Privacidade e Analise de Fairness\n",
    "\n",
    "**Objetivo:** garantir conformidade com principios de ethical AI.\n",
    "\n",
    "1. **Pseudonimizacao**: `Customer_ID` substituido por hash SHA-256 truncado\n",
    "2. **Minimizacao**: remocao de colunas com PII direto (email, telefone, nome)\n",
    "3. **Generalizacao**: idade exata substituida por faixa etaria\n",
    "4. **Fairness**: analise de disparidade de churn rate por atributo sensivel\n",
]))

cells.append(co([
    "from src.privacy.anonymization import anonymize_dataframe, minimization_report\n",
    "\n",
    "df_raw = pd.read_csv('../data/raw/ecommerce_customer_churn_dataset.csv')\n",
    "df_anon = anonymize_dataframe(df_raw, id_col='Customer_ID')\n",
    "\n",
    "rep = minimization_report(df_raw, df_anon)\n",
    "print(f'Colunas originais:    {rep[\"original_columns\"]}')\n",
    "print(f'Colunas anonimizadas: {rep[\"anonymized_columns\"]}')\n",
    "print(f'Removidas:  {rep[\"removed\"] or \"nenhuma\"}')\n",
    "print(f'Adicionadas:{rep[\"added\"] or \"nenhuma\"}')\n",
    "print()\n",
    "print('Exemplo de Customer_ID anonimizado:')\n",
    "print(df_anon[['Customer_ID']].head(3).to_string(index=False))\n",
]))

cells.append(co([
    "from src.privacy.bias_check import check_bias, print_bias_report\n",
    "\n",
    "df_check = df.copy()\n",
    "# Restaurar colunas sensiveis do raw para a analise\n",
    "for col in ['Gender', 'Country']:\n",
    "    if col in df_raw.columns:\n",
    "        df_check[col] = df_raw[col].values\n",
    "\n",
    "result = check_bias(df_check, sensitive_cols=['Gender', 'Country'])\n",
    "print_bias_report(result)\n",
]))

# ------------------------------------------------------------------
# GRAVAR
# ------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"main.ipynb gravado em {OUT}")
print(f"Total de celulas: {len(cells)}")
