"""Robustness Testing (V3) — testes de robustez do modelo de churn, no estilo
empresarial (não FGSM acadêmico): perturbação de features, missing-data
attacks, distribution shift e detecção de out-of-distribution.

Consolidado em `robustness_report`, cujo resultado alimenta o *robustness
gate* do Argus via o `ModelSecurityReport` do ThemisAI.
"""
from __future__ import annotations

from src.robustness.distribution_shift import covariate_shift, psi
from src.robustness.missing_data import missing_data_impact, missing_data_sweep
from src.robustness.ood import OODDetector
from src.robustness.perturbation import feature_perturbation, fragility_score
from src.robustness.report import robustness_report

__all__ = [
    "feature_perturbation",
    "fragility_score",
    "missing_data_impact",
    "missing_data_sweep",
    "covariate_shift",
    "psi",
    "OODDetector",
    "robustness_report",
]
