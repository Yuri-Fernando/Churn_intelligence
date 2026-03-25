"""
Verificacao rapida de fairness/bias para uso no pipeline.

Funcoes utilitarias que complementam bias_analysis.py com
verificacoes diretas e alertas de disparidade.
"""

import pandas as pd
from src.privacy.bias_analysis import analyze_bias


DISPARITY_THRESHOLD = 0.15  # diferenca maxima aceitavel de churn rate entre grupos


def check_bias(df: pd.DataFrame, sensitive_cols=None, threshold: float = DISPARITY_THRESHOLD) -> dict:
    """
    Executa analise de bias e retorna alertas para grupos com disparidade alta.

    Returns:
        dict com keys:
          'report'  -> taxa de churn por grupo (raw)
          'alerts'  -> lista de alertas de disparidade
          'passed'  -> True se nenhum grupo excedeu o threshold
    """
    if sensitive_cols is None:
        sensitive_cols = ["Gender", "Country"]

    report = analyze_bias(df, sensitive_cols=sensitive_cols)
    alerts = []

    for col, group_rates in report.items():
        values = list(group_rates.values())
        if not values:
            continue
        disparity = max(values) - min(values)
        if disparity > threshold:
            alerts.append({
                "column": col,
                "disparity": round(disparity, 4),
                "max_group": max(group_rates, key=group_rates.get),
                "min_group": min(group_rates, key=group_rates.get),
            })

    return {
        "report": report,
        "alerts": alerts,
        "passed": len(alerts) == 0,
    }


def print_bias_report(result: dict) -> None:
    """Imprime relatorio de bias de forma legivel."""
    print("=== Relatorio de Fairness ===")
    for col, rates in result["report"].items():
        print(f"\n{col}:")
        for group, rate in sorted(rates.items(), key=lambda x: -x[1]):
            print(f"  {group}: {rate:.3f} ({rate*100:.1f}% churn)")

    if result["passed"]:
        print("\n[OK] Nenhuma disparidade significativa detectada.")
    else:
        print(f"\n[ALERTA] {len(result['alerts'])} disparidade(s) acima do threshold:")
        for a in result["alerts"]:
            print(f"  - {a['column']}: {a['disparity']:.3f} "
                  f"(max: {a['max_group']}, min: {a['min_group']})")
