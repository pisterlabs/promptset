from services.ai_service import AIService
import requests
import io
import openai
import os


class OpenAIService(AIService):
    def __init__(self):
        # we handle all the api config directly in the calls below
        pass

    def run_llm(self, messages, stream=True):
        model = os.getenv("OPEN_AI_MODEL")
        if not model:
            model = "gpt-4"
        response = openai.ChatCompletion.create(
            api_type='openai',
            api_version='2020-11-07',
            api_base="https://api.openai.com/v1",
            api_key=os.getenv("OPEN_AI_KEY"),
            model=model,
            stream=stream,
            messages=messages
        )

        return response
