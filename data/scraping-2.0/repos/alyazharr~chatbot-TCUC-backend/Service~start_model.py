import os
import openai
import io
from dotenv import load_dotenv
from openai import OpenAI

class openaiCallAPI:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def get_env():
        load_dotenv()
        org_key = os.environ.get("OPENAI_ORGANIZATION", None)
        api_key = os.environ.get("OPENAI_API_KEY", None)
        access_key = os.environ.get("ACCESS_KEY", None)

        return {"org_key" : org_key,
                "api_key" : api_key,
                "access_key" : access_key}

    def get_completion_from_messages(self, messages, model="gpt-3.5-turbo", temperature=0.9, max_tokens=500):
        completion = self.client.chat.completions.create(model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens)
        return completion.choices[0].message.content
