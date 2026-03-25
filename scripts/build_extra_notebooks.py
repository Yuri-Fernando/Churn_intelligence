"""Gera experiments.ipynb e benchmarks.ipynb."""
import json, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": [src] if isinstance(src, str) else src}

def co(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [src] if isinstance(src, str) else src}

def save(cells, name):
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.10.0"}},
        "cells": cells,
    }
    path = os.path.join(ROOT, "notebooks", name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Salvo: {path} ({len(cells)} celulas)")


# ==========================================================
# EXPERIMENTS.IPYNB — Comparacao de modelos + tuning
# ==========================================================
exp = []

exp.append(md([
    "# Experiments — Comparacao de Modelos e Hyperparameter Tuning\n",
    "\n",
    "Avaliacao sistematica dos tres algoritmos com cross-validation e grid search.\n",
    "\n",
    "| Secao | Conteudo |\n",
    "|-------|---------|\n",
    "| 1 | Cross-validation 5-fold em todos os modelos |\n",
    "| 2 | GridSearchCV no RandomForest |\n",
    "| 3 | Curvas ROC comparativas |\n",
    "| 4 | Matriz de confusao |\n",
    "| 5 | MLflow tracking dos experimentos |\n",
]))

exp.append(md("## Setup\n"))
exp.append(co([
    "import sys, os, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import mlflow\n",
    "from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n",
    "from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from src.features.build_features import build_features\n",
    "from src.models.model_utils import split_features_target\n",
    "\n",
    "df = build_features(\n",
    "    input_csv='../data/raw/ecommerce_customer_churn_dataset.csv',\n",
    "    output_csv='../data/processed/features.csv',\n",
    ")\n",
    "X_train, X_test, y_train, y_test = split_features_target(df)\n",
    "print(f'Train: {len(X_train):,} | Test: {len(X_test):,}')\n",
]))

exp.append(md("## 1 — Cross-Validation 5-fold\n"))
exp.append(co([
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "\n",
    "models = {\n",
    "    'LogisticRegression':  LogisticRegression(max_iter=1000),\n",
    "    'RandomForest':        RandomForestClassifier(n_estimators=100, random_state=42),\n",
    "    'GradientBoosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),\n",
    "}\n",
    "\n",
    "cv_results = {}\n",
    "for name, model in models.items():\n",
    "    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)\n",
    "    cv_results[name] = scores\n",
    "    print(f'{name:25s}  AUC={scores.mean():.4f} (+/- {scores.std():.4f})')\n",
]))

exp.append(md("## 2 — GridSearchCV no RandomForest\n"))
exp.append(co([
    "param_grid = {\n",
    "    'n_estimators': [50, 100, 200],\n",
    "    'max_depth':    [None, 10, 20],\n",
    "    'min_samples_split': [2, 5],\n",
    "}\n",
    "\n",
    "gs = GridSearchCV(\n",
    "    RandomForestClassifier(random_state=42),\n",
    "    param_grid,\n",
    "    cv=3,\n",
    "    scoring='roc_auc',\n",
    "    n_jobs=-1,\n",
    "    verbose=1,\n",
    ")\n",
    "gs.fit(X_train, y_train)\n",
    "\n",
    "print(f'Melhores parametros: {gs.best_params_}')\n",
    "print(f'Melhor AUC (CV):     {gs.best_score_:.4f}')\n",
    "best_rf = gs.best_estimator_\n",
]))

exp.append(md("## 3 — Curvas ROC comparativas\n"))
exp.append(co([
    "fig, ax = plt.subplots(figsize=(8, 6))\n",
    "\n",
    "for name, model in models.items():\n",
    "    model.fit(X_train, y_train)\n",
    "    proba = model.predict_proba(X_test)[:, 1]\n",
    "    fpr, tpr, _ = roc_curve(y_test, proba)\n",
    "    auc = roc_auc_score(y_test, proba)\n",
    "    ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')\n",
    "\n",
    "# Melhor RF (GridSearch)\n",
    "proba_best = best_rf.predict_proba(X_test)[:, 1]\n",
    "fpr_b, tpr_b, _ = roc_curve(y_test, proba_best)\n",
    "auc_b = roc_auc_score(y_test, proba_best)\n",
    "ax.plot(fpr_b, tpr_b, '--', label=f'RF Tuned (AUC={auc_b:.3f})')\n",
    "\n",
    "ax.plot([0,1],[0,1],'k:', alpha=0.4)\n",
    "ax.set_xlabel('FPR')\n",
    "ax.set_ylabel('TPR')\n",
    "ax.set_title('Curvas ROC — Comparacao de Modelos')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
]))

exp.append(md("## 4 — Matriz de Confusao (Melhor Modelo)\n"))
exp.append(co([
    "from sklearn.metrics import ConfusionMatrixDisplay\n",
    "\n",
    "y_pred = best_rf.predict(X_test)\n",
    "print(classification_report(y_test, y_pred, target_names=['Nao churn', 'Churn']))\n",
    "\n",
    "ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Nao churn', 'Churn'])\n",
    "plt.title('Matriz de Confusao — RandomForest Tuned')\n",
    "plt.show()\n",
]))

exp.append(md("## 5 — MLflow Tracking dos Experimentos\n"))
exp.append(co([
    "mlflow.set_tracking_uri('sqlite:///../notebooks/mlflow.db')\n",
    "mlflow.set_experiment('experiments_comparacao')\n",
    "\n",
    "for name, model in {**models, 'RF_Tuned': best_rf}.items():\n",
    "    if name != 'RF_Tuned':\n",
    "        model.fit(X_train, y_train)\n",
    "    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])\n",
    "    prec = model.score(X_test, y_test)\n",
    "\n",
    "    with mlflow.start_run(run_name=name):\n",
    "        mlflow.log_metric('roc_auc', auc)\n",
    "        mlflow.log_param('model_type', name)\n",
    "        if hasattr(model, 'n_estimators'):\n",
    "            mlflow.log_param('n_estimators', model.n_estimators)\n",
    "        print(f'{name:25s}  AUC={auc:.4f}')\n",
    "\n",
    "print('\\nExperimentos registrados no MLflow.')\n",
]))

save(exp, "experiments.ipynb")


# ==========================================================
# BENCHMARKS.IPYNB — Latencia e throughput da API
# ==========================================================
bench = []

bench.append(md([
    "# Benchmarks — Latencia e Throughput\n",
    "\n",
    "Mede o desempenho do pipeline em diferentes cenarios:\n",
    "\n",
    "| Secao | Conteudo |\n",
    "|-------|---------|\n",
    "| 1 | Latencia de predicao individual (modelo) |\n",
    "| 2 | Throughput batch (predicoes/segundo) |\n",
    "| 3 | Benchmark do pipeline cognitivo completo |\n",
    "| 4 | Benchmark do streaming processor |\n",
]))

bench.append(md("## Setup\n"))
bench.append(co([
    "import sys, os, time, warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import joblib\n",
    "\n",
    "from src.features.build_features import build_features\n",
    "from src.models.model_utils import split_features_target\n",
    "\n",
    "df = build_features(\n",
    "    input_csv='../data/raw/ecommerce_customer_churn_dataset.csv',\n",
    "    output_csv='../data/processed/features.csv',\n",
    ")\n",
    "X_train, X_test, y_train, y_test = split_features_target(df)\n",
    "rf_model = joblib.load('../models/rf_model.pkl')\n",
    "print('Setup concluido')\n",
]))

bench.append(md("## 1 — Latencia de Predicao Individual\n"))
bench.append(co([
    "N_RUNS = 1000\n",
    "single_row = X_test.iloc[[0]]\n",
    "\n",
    "# Warm-up\n",
    "for _ in range(10):\n",
    "    rf_model.predict_proba(single_row)\n",
    "\n",
    "times = []\n",
    "for _ in range(N_RUNS):\n",
    "    t0 = time.perf_counter()\n",
    "    rf_model.predict_proba(single_row)\n",
    "    times.append((time.perf_counter() - t0) * 1000)\n",
    "\n",
    "times = np.array(times)\n",
    "print(f'Predicao individual ({N_RUNS} runs):')\n",
    "print(f'  Media:   {times.mean():.3f} ms')\n",
    "print(f'  Mediana: {np.median(times):.3f} ms')\n",
    "print(f'  P95:     {np.percentile(times, 95):.3f} ms')\n",
    "print(f'  P99:     {np.percentile(times, 99):.3f} ms')\n",
    "print(f'  Max:     {times.max():.3f} ms')\n",
]))

bench.append(md("## 2 — Throughput Batch\n"))
bench.append(co([
    "batch_sizes = [1, 10, 100, 500, 1000, 5000]\n",
    "results = []\n",
    "\n",
    "for bs in batch_sizes:\n",
    "    batch = X_test.iloc[:bs]\n",
    "    t0 = time.perf_counter()\n",
    "    rf_model.predict_proba(batch)\n",
    "    elapsed = time.perf_counter() - t0\n",
    "    throughput = bs / elapsed\n",
    "    results.append({'batch_size': bs, 'latency_ms': elapsed*1000, 'throughput_ps': throughput})\n",
    "    print(f'Batch={bs:5d}  {elapsed*1000:7.2f}ms  {throughput:,.0f} pred/s')\n",
    "\n",
    "df_bench = pd.DataFrame(results)\n",
    "display(df_bench.round(2))\n",
]))

bench.append(md("## 3 — Pipeline Cognitivo Completo\n"))
bench.append(co([
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
    "sample = X_test.head(50)\n",
    "scores = rf_model.predict_proba(sample)[:, 1]\n",
    "\n",
    "N = len(sample)\n",
    "t0 = time.perf_counter()\n",
    "for i, (_, row) in enumerate(sample.iterrows()):\n",
    "    feats = row.to_dict()\n",
    "    score = float(scores[i])\n",
    "    analysis  = analyst.run(score, feats, '')\n",
    "    decision  = strategy.run(analysis, 'offer_discount')\n",
    "    explanation = rag.run(f'u_{i}', score, feats)\n",
    "    state = auditor.run({'churn_score': score, 'strategy': decision})\n",
    "\n",
    "elapsed = time.perf_counter() - t0\n",
    "print(f'Pipeline cognitivo em {N} usuarios:')\n",
    "print(f'  Total:    {elapsed*1000:.1f} ms')\n",
    "print(f'  Media:    {elapsed/N*1000:.2f} ms/usuario')\n",
    "print(f'  Throughput: {N/elapsed:.0f} usuarios/s')\n",
]))

bench.append(md("## 4 — Streaming Processor\n"))
bench.append(co([
    "from src.streaming.event_processor import StreamProcessor, generate_synthetic_events\n",
    "\n",
    "events = generate_synthetic_events(n=100, seed=0)\n",
    "processor = StreamProcessor(model_path='../models/rf_model.pkl')\n",
    "\n",
    "t0 = time.perf_counter()\n",
    "results = processor.run(events, delay=0.0, verbose=False)\n",
    "total_ms = (time.perf_counter() - t0) * 1000\n",
    "\n",
    "summary = processor.summary()\n",
    "print(f'Streaming benchmark ({len(events)} eventos):')\n",
    "print(f'  Tempo total:     {total_ms:.1f} ms')\n",
    "print(f'  Latencia media:  {summary[\"latency_mean_ms\"]} ms/evento')\n",
    "print(f'  Latencia max:    {summary[\"latency_max_ms\"]} ms')\n",
    "print(f'  Throughput:      {len(events)/total_ms*1000:.0f} eventos/s')\n",
    "print(f'  Score medio:     {summary[\"score_mean\"]}')\n",
    "print(f'  Segmentos:       {summary[\"segments\"]}')\n",
]))

save(bench, "benchmarks.ipynb")
print("Notebooks gerados.")
