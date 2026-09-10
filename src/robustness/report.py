"""Robustness report — consolida os testes de robustez do modelo de churn
num único dicionário/texto (equivalente empresarial do MODEL SECURITY
REPORT do ThemisAI, para features tabulares de negócio).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.robustness.distribution_shift import covariate_shift
from src.robustness.missing_data import missing_data_sweep
from src.robustness.ood import OODDetector
from src.robustness.perturbation import feature_perturbation, fragility_score


def robustness_report(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test,
    key_features: list[str] | None = None,
    threshold: float = 0.5,
) -> dict:
    num_cols = list(X_test.select_dtypes(include=[np.number]).columns)
    key_features = key_features or num_cols[:6]

    frag = fragility_score(model, X_test, key_features, threshold)
    perturb = {f: feature_perturbation(model, X_test, f, deltas=[-1, 1], threshold=threshold)
               for f in key_features}
    missing = missing_data_sweep(model, X_test, key_features, threshold)[:5]

    shift_feature = key_features[0]
    shift = covariate_shift(model, X_test, y_test, shift_feature, threshold=threshold)
    worst_shift_drop = max(r["accuracy_drop"] for r in shift)

    ood = OODDetector().fit(X_train)
    ood_flag_rate = ood.flag_rate(X_test)

    if frag > 0.15 or worst_shift_drop > 0.15:
        risk = "HIGH"
    elif frag > 0.05 or worst_shift_drop > 0.07:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    report = {
        "fragility_score": frag,
        "perturbation": perturb,
        "top_missing_data_dependencies": missing,
        "covariate_shift": {"feature": shift_feature, "worst_accuracy_drop": worst_shift_drop, "curve": shift},
        "ood_flag_rate": ood_flag_rate,
        "robustness_risk": risk,
        "recommendations": _recommend(risk, frag, worst_shift_drop, ood_flag_rate),
    }
    report["rendered"] = _render(report)
    return report


def _recommend(risk, frag, shift_drop, ood_rate) -> list[str]:
    recs = []
    if frag > 0.05:
        recs.append("adicionar perturbação de features ao treino (data augmentation) para suavizar a fronteira")
    if shift_drop > 0.07:
        recs.append("monitorar PSI/KS das features em produção e disparar re-treino em drift")
    if ood_rate > 0.05:
        recs.append("rotear entradas OOD para revisão humana em vez de decisão automática")
    if not recs:
        recs.append("robustez dentro do aceitável; manter monitoramento padrão")
    return recs


def _render(r: dict) -> str:
    lines = ["=" * 48, "  MODEL ROBUSTNESS REPORT — churn", "=" * 48, ""]
    lines.append(f"Fragility score (±1 em features-chave) .. {r['fragility_score']:.3f}")
    lines.append(f"Pior queda de acurácia sob shift ....... {r['covariate_shift']['worst_accuracy_drop']:.3f}")
    lines.append(f"Taxa de entradas OOD .................. {r['ood_flag_rate']:.1%}")
    lines.append(f"Risco de robustez .................... {r['robustness_risk']}")
    lines.append("")
    lines.append("Dependências de missing-data (top):")
    for m in r["top_missing_data_dependencies"]:
        lines.append(f"  - {m['feature']:<24} flip {m['decision_flip_rate']:.1%}")
    lines.append("")
    lines.append("Mitigações recomendadas:")
    for rec in r["recommendations"]:
        lines.append(f"  - {rec}")
    lines.append("=" * 48)
    return "\n".join(lines)
