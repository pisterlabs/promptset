import openai

from config import OPENAI_AI_API_KEY


class GPTClient:
    def __init__(self):
        """Initialize the GPT client with the OpenAI API key."""
        openai.api_key = OPENAI_AI_API_KEY

    def get_response_from_llm(self, model: str, prompt: str):
        """Get a response from a specified language model."""
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
            temperature=0.5,
        )
        return response["choices"][0]["message"]["content"]


