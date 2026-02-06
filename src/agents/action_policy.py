# Lógica de decisão de ações
def action_for_score(score: float):
    if score < 0.3:
        return "recommend_new_product"
    elif score < 0.7:
        return "no_action"
    else:
        return "offer_discount"
