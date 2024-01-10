import os

from kani.engines.openai import OpenAIEngine

CHAT_HYPERPARAMS = dict(
    model="gpt-4",
    temperature=1,
    top_p=0.95,
    frequency_penalty=0.3,
)

api_key = os.getenv("OPENAI_API_KEY")
chat_engine = OpenAIEngine(api_key=api_key, **CHAT_HYPERPARAMS)
long_engine = OpenAIEngine(api_key=api_key, model="gpt-4-32k", temperature=0.1)
