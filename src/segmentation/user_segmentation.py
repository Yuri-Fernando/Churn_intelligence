import pandas as pd
import numpy as np

def segment_users(df):
    """
    Segmenta usuários em:
      - Engajados
      - Em risco
      - Churned (já pararam)
    
    Usa features simples: recência, frequência, intensidade e churn.
    """
    df = df.copy()
    
    conditions = [
        (df['churn'] == 1),  # já churned
        (df['recency_days'] > 60) & (df['frequency'] < 5),  # risco
        (df['recency_days'] <= 60) & (df['frequency'] >= 5)  # engajado
    ]
    
    choices = ['churned', 'at_risk', 'engaged']
    
    df['segment'] = np.select(conditions, choices, default='neutral')
    
    return df
