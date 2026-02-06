# src/features/build_features.py

import pandas as pd
import numpy as np

def build_features(
    input_csv="../data/raw/ecommerce_customer_churn_dataset.csv",
    output_csv="../data/processed/features.csv",
    churn_days=30
):
    """
    Constrói features a partir do dataset de churn de e-commerce.
    Gera recência, frequência, duração média, intensidade de uso,
    codifica variáveis categóricas e define churn.
    """

    # -----------------------------
    # 1️⃣ Carrega dados
    # -----------------------------
    data = pd.read_csv(input_csv)

    # -----------------------------
    # 2️⃣ Features numéricas básicas
    # -----------------------------
    data["recency_days"] = data.get("Days_Since_Last_Purchase", 0)
    data["frequency"] = data.get("Login_Frequency", 0)
    data["avg_session_duration"] = data.get("Session_Duration_Avg", 0)
    data["intensity"] = data.get("Pages_Per_Session", 0)
    data["engagement_trend"] = 0  # placeholder simples

    # Target
    data["churn"] = data.get("Churned", 0)

    # -----------------------------
    # 3️⃣ Codificação de variáveis categóricas
    # -----------------------------
    categorical_cols = ["Gender", "Country", "City", "Payment_Method_Diversity", "Signup_Quarter"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype(str)  # garante string
            data[col] = pd.factorize(data[col])[0]  # transforma em números

    # -----------------------------
    # 4️⃣ Preenche NaNs
    # -----------------------------
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            data[col] = data[col].fillna(data[col].median())  # numéricos -> mediana
        else:
            data[col] = data[col].fillna(0)  # categóricos -> 0

    # -----------------------------
    # 5️⃣ Salva features
    # -----------------------------
    data.to_csv(output_csv, index=False)
    print(f"Features geradas e salvas em {output_csv}")

    return data
