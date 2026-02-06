# src/evaluation/shap_analysis.py

import shap
import matplotlib.pyplot as plt

def explain_model(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train)
