import os
from os.path import join, dirname

from dotenv import load_dotenv
from langchain.llms import OpenAI

dotenv_path = join(dirname(__file__), "./../.env")
load_dotenv(dotenv_path)


def get_gpt_completions(prompt, max_tokens=100, model="gpt-4"):
    return OpenAI(model_name=model, max_tokens=max_tokens,
                  openai_api_key=os.environ.get("OPENAI_API_KEY"))(prompt)
