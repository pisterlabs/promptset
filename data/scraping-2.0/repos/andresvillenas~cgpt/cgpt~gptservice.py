import openai

from config.settings import Config


class GptService:
    """
    Service for interacting with the OpenAI GPT-3 API
    """

    def __init__(self):
        config = Config()
        openai.api_key = config.openai.api_key

        self.chat_model = "gpt-3.5-turbo"

    def get_response(self, prompt, temperature=0.25, max_tokens=1000):
        response = openai.ChatCompletion.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You are an expert about operating system commands"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        reply = response.choices[0].message['content']

        return reply.strip()
