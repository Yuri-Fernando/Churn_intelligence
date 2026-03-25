"""
Dashboard executivo de Churn Intelligence.

Uso:
  streamlit run src/dashboard/app.py

Mostra:
  - KPIs de churn (taxa, distribuicao de scores, alto risco)
  - Segmentacao de usuarios com distribuicao visual
  - Analise de fairness por genero e pais
  - Top features de risco (SHAP)
  - Tabela de decisoes dos agentes
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------
# Config da pagina
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
)

# -----------------------------------------------------------------------
# Carregamento de dados (cache)
# -----------------------------------------------------------------------
@st.cache_data
def load_data():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    features_path = os.path.join(base, "data/processed/features.csv")
    raw_path = os.path.join(base, "data/raw/ecommerce_customer_churn_dataset.csv")
    df = pd.read_csv(features_path)
    df_raw = pd.read_csv(raw_path)
    return df, df_raw


@st.cache_resource
def load_model():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_path = os.path.join(base, "models/rf_model.pkl")
    return joblib.load(model_path)


@st.cache_data
def compute_scores(_model, _df):
    from src.models.model_utils import split_features_target
    X_train, X_test, y_train, y_test = split_features_target(_df)
    scores = _model.predict_proba(X_test)[:, 1]
    df_out = X_test.copy()
    df_out["churn_score"] = scores
    df_out["y_true"] = y_test.values
    return df_out, scores


# -----------------------------------------------------------------------
# Layout principal
# -----------------------------------------------------------------------
st.title("Churn Intelligence Dashboard")
st.caption("Pipeline de predicao de churn com segmentacao dinamica e agentes cognitivos.")

try:
    df, df_raw = load_data()
    model = load_model()
    df_scores, scores = compute_scores(model, df)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# -----------------------------------------------------------------------
# KPIs
# -----------------------------------------------------------------------
st.header("KPIs Globais")

col1, col2, col3, col4 = st.columns(4)

churn_rate      = df["churn"].mean()
high_risk       = (scores >= 0.7).mean()
medium_risk     = ((scores >= 0.4) & (scores < 0.7)).mean()
score_mean      = scores.mean()

col1.metric("Taxa de Churn (real)", f"{churn_rate:.1%}")
col2.metric("Alto Risco (score > 0.7)", f"{high_risk:.1%}")
col3.metric("Risco Medio (0.4-0.7)", f"{medium_risk:.1%}")
col4.metric("Score Medio de Churn", f"{score_mean:.3f}")

# -----------------------------------------------------------------------
# Distribuicao de scores
# -----------------------------------------------------------------------
st.header("Distribuicao de Scores de Churn")

import plotly.express as px
import plotly.graph_objects as go

fig_hist = px.histogram(
    x=scores,
    nbins=40,
    labels={"x": "Churn Score"},
    color_discrete_sequence=["#e74c3c"],
    title="Distribuicao de Scores",
)
fig_hist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Risco medio")
fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red",    annotation_text="Alto risco")
fig_hist.update_layout(showlegend=False)
st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------------------------------------------------
# Segmentacao
# -----------------------------------------------------------------------
st.header("Segmentacao de Usuarios")

col_a, col_b = st.columns(2)

# Segmentacao por score
seg_labels = pd.cut(
    scores,
    bins=[-0.01, 0.4, 0.7, 1.01],
    labels=["Engajado", "Ocasional", "Alto Risco"],
)
seg_counts = seg_labels.value_counts().reset_index()
seg_counts.columns = ["Segmento", "Usuarios"]

fig_pie = px.pie(
    seg_counts,
    values="Usuarios",
    names="Segmento",
    title="Distribuicao por Segmento",
    color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
)
col_a.plotly_chart(fig_pie, use_container_width=True)

# Box plot por segmento
fig_box = px.box(
    x=seg_labels,
    y=scores,
    labels={"x": "Segmento", "y": "Churn Score"},
    title="Score por Segmento",
    color=seg_labels,
    color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
)
col_b.plotly_chart(fig_box, use_container_width=True)

# -----------------------------------------------------------------------
# Fairness
# -----------------------------------------------------------------------
st.header("Analise de Fairness")

sensitive_cols = [c for c in ["Gender", "Country"] if c in df_raw.columns]

if sensitive_cols:
    df_fair = df.copy()
    for col in sensitive_cols:
        df_fair[col] = df_raw[col].values

    tabs = st.tabs(sensitive_cols)
    for tab, col in zip(tabs, sensitive_cols):
        with tab:
            group_churn = df_fair.groupby(col)["churn"].mean().sort_values(ascending=False)
            fig_fair = px.bar(
                x=group_churn.index,
                y=group_churn.values,
                labels={"x": col, "y": "Taxa de Churn"},
                title=f"Taxa de Churn por {col}",
                color=group_churn.values,
                color_continuous_scale="RdYlGn_r",
            )
            fig_fair.update_coloraxes(showscale=False)
            tab.plotly_chart(fig_fair, use_container_width=True)

            disparity = group_churn.max() - group_churn.min()
            if disparity > 0.15:
                tab.warning(f"Disparidade de {disparity:.2%} detectada em {col}.")
            else:
                tab.success(f"Disparidade de {disparity:.2%} — dentro do limite aceitavel.")

# -----------------------------------------------------------------------
# Feature importance
# -----------------------------------------------------------------------
st.header("Importancia de Features (Random Forest)")

if hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
    importances = pd.Series(
        model.feature_importances_,
        index=model.feature_names_in_,
    ).sort_values(ascending=True).tail(15)

    fig_imp = px.bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        labels={"x": "Importancia", "y": "Feature"},
        title="Top 15 Features",
        color=importances.values,
        color_continuous_scale="Blues",
    )
    fig_imp.update_coloraxes(showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

# -----------------------------------------------------------------------
# Decisoes dos agentes
# -----------------------------------------------------------------------
st.header("Decisoes dos Agentes Cognitivos")

decisions_path = os.path.join(
    os.path.dirname(__file__), "../../notebooks/agent_decision_summary.csv"
)
if os.path.exists(decisions_path):
    df_dec = pd.read_csv(decisions_path)
    st.dataframe(df_dec, use_container_width=True)

    if "action" in df_dec.columns:
        action_counts = df_dec["action"].value_counts().reset_index()
        action_counts.columns = ["Acao", "Quantidade"]
        fig_act = px.bar(
            action_counts,
            x="Acao",
            y="Quantidade",
            title="Distribuicao de Acoes Recomendadas",
            color="Acao",
        )
        st.plotly_chart(fig_act, use_container_width=True)
else:
    st.info("Execute o notebook (Fase 3) para gerar agent_decision_summary.csv.")

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.divider()
st.caption("Churn Intelligence Pipeline | MLflow + FastAPI + Agentes Cognitivos + RAG")
