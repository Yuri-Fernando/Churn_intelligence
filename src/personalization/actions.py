def generate_action(segment):
    """
    Retorna ação recomendada baseada no segmento.
    """
    if segment == 'at_risk':
        return 'send_discount_offer'
    elif segment == 'engaged':
        return 'recommend_new_product'
    elif segment == 'churned':
        return 're-engagement_email'
    else:
        return 'no_action'
