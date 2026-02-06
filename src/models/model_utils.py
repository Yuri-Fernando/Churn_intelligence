# src/models/model_utils.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score

def split_features_target(df, target_col="churn", test_size=0.2, random_state=42):
    """
    Divide dataframe em X e y, ignorando 'user_id' caso não exista.
    """
    X = df.drop(columns=[target_col, "user_id"], errors='ignore')
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    """
    Avalia modelo retornando ROC-AUC, Precision e Recall.
    """
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_prob = y_pred  # fallback se predict_proba não existir
    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0)
    }
