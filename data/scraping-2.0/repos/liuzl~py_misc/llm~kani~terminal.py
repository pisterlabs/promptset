# encoding: UTF-8
import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

api_key = os.getenv("API_KEY")
api_base = os.getenv("API_BASE")

from kani import Kani, chat_in_terminal
from kani.engines.openai import OpenAIEngine

engine = OpenAIEngine(api_base=api_base, api_key=api_key, model="gpt-3.5-turbo")
ai = Kani(engine)
chat_in_terminal(ai)
