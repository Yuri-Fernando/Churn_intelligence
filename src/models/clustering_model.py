from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


CLUSTER_FEATURES = [
    "recency_days", "frequency", "avg_session_duration", "intensity"
]

CLUSTER_NAMES = {
    0: "engaged",
    1: "occasional",
    2: "at_risk",
    3: "inactive",
}


def cluster_users(df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    """
    Segmenta usuarios via KMeans usando features comportamentais.

    Retorna df com coluna 'cluster_id' (int) e 'cluster_label' (string).
    """
    available = [c for c in CLUSTER_FEATURES if c in df.columns]
    if not available:
        raise ValueError(f"Nenhuma feature de clustering encontrada. Esperado: {CLUSTER_FEATURES}")

    X = df[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X_scaled)

    result = df.copy()
    result["cluster_id"] = labels

    # Nomear clusters por churn_rate media (mais alto = maior risco)
    if "churn" in df.columns:
        cluster_churn = result.groupby("cluster_id")["churn"].mean().sort_values(ascending=False)
        rank_map = {cid: i for i, cid in enumerate(cluster_churn.index)}
        result["cluster_id"] = result["cluster_id"].map(rank_map)

    result["cluster_label"] = result["cluster_id"].map(CLUSTER_NAMES).fillna("unknown")
    return result


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna estatisticas por cluster."""
    cols = ["cluster_label"] + [c for c in CLUSTER_FEATURES if c in df.columns]
    if "churn" in df.columns:
        cols.append("churn")
    return df[cols].groupby("cluster_label").mean().round(3)
