# Wrapper para LLMs
from langchain.chat_models import ChatOpenAI

class LLMWrapper:
    def __init__(self, model_name="gpt-4", temperature=0.7):
        self.model = ChatOpenAI(model_name=model_name, temperature=temperature)

    def generate_text(self, prompt: str):
        return self.model.predict(prompt)
