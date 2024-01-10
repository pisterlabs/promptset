from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class Client():
    MODEL_TEXT_DAVINCI = "text-davinci-003"
    MODEL_GPT_35 = "gpt-3.5-turbo"
    MODEL_GPT_4 = "gpt-4-1106-preview"

    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
