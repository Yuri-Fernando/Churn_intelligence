# Exemplo de integração com OpenAI GPT
from openai import OpenAI

client = OpenAI()

def generate_recommendation(prompt: str, max_tokens=100):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content
