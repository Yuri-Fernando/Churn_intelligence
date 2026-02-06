def log_decision(user_id, prompt, context, response):
    print({
        "user": user_id,
        "prompt": prompt,
        "context": context,
        "response": response
    })
