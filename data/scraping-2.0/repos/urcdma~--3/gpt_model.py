import openai
from typing import Optional

class GPTModel:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the GPT Da Vinci model with the provided OpenAI API key
        """
        self.api_key = api_key if api_key else 'key'
        openai.api_key = self.api_key

    def interpret_knowledge_point(self, knowledge_point: str) -> str:
        """
        Use the GPT Da Vinci model to interpret the knowledge point
        """
        response = openai.Completion.create(
            engine="davinci",
            prompt=knowledge_point,
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].text.strip()
