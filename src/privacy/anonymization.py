"""
Pseudonimizacao e minimizacao de dados.

Principios aplicados:
  - Pseudonimizacao: user_id substituido por hash SHA-256
  - Minimizacao: remocao de colunas com PII direto
  - Generalizacao: faixas etarias ao inves de idade exata
"""

import hashlib
import pandas as pd

PII_COLUMNS = ["Email", "Phone", "Name", "CPF", "full_name", "email", "phone"]

AGE_BINS = [0, 18, 25, 35, 45, 60, 100]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-59", "60+"]


def pseudonymize_id(user_id: str, salt: str = "churn_pipeline") -> str:
    """Retorna hash SHA-256 truncado do user_id."""
    raw = f"{salt}:{user_id}".encode("utf-8")
    return "u_" + hashlib.sha256(raw).hexdigest()[:16]


def anonymize_dataframe(df: pd.DataFrame, id_col: str = "Customer_ID") -> pd.DataFrame:
    """
    Aplica pseudonimizacao e minimizacao ao DataFrame.

    1. Pseudonimiza id_col com SHA-256
    2. Remove colunas PII diretas
    3. Generaliza coluna de idade (se existir)
    """
    result = df.copy()

    if id_col in result.columns:
        result[id_col] = result[id_col].astype(str).apply(pseudonymize_id)

    cols_to_drop = [c for c in PII_COLUMNS if c in result.columns]
    if cols_to_drop:
        result.drop(columns=cols_to_drop, inplace=True)

    if "Age" in result.columns:
        result["age_group"] = pd.cut(
            result["Age"],
            bins=AGE_BINS,
            labels=AGE_LABELS,
            right=False,
        ).astype(str)
        result.drop(columns=["Age"], inplace=True)

    return result


def minimization_report(df_original: pd.DataFrame, df_anonymized: pd.DataFrame) -> dict:
    """Relatorio comparando colunas antes e depois da anonimizacao."""
    removed = set(df_original.columns) - set(df_anonymized.columns)
    added = set(df_anonymized.columns) - set(df_original.columns)
    return {
        "original_columns": len(df_original.columns),
        "anonymized_columns": len(df_anonymized.columns),
        "removed": sorted(removed),
        "added": sorted(added),
    }
