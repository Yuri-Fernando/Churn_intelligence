# src/privacy/bias_analysis.py
import pandas as pd

def analyze_bias(df, sensitive_cols=["Gender", "Country"]):
    """
    Analisa fairness / bias simples em churn.
    Retorna um dicionário com a média de churn por grupo sensível.
    """
    report = {}
    for col in sensitive_cols:
        if col in df.columns:
            counts = df.groupby(col)["churn"].mean()
            report[col] = counts.to_dict()
    return report
