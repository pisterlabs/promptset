# quantum_leisure.py

from openai_api import generate_text

class QuantumLeisure:
    def __init__(self):
        self.prompt = "Quantum Leisure and Recreation: "

    def get_leisure_idea(self, additional_prompt="", max_tokens=100):
        """
        Function to generate a leisure idea using OpenAI's GPT-3 model.
        """
        full_prompt = self.prompt + additional_prompt
        leisure_idea = generate_text(full_prompt, max_tokens)
        return leisure_idea
