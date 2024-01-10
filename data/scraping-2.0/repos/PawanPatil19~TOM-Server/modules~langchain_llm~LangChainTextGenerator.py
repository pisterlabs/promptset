import json
import os
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from config import OPENAI_CREDENTIAL_FILE

KEY_OPENAI_API = 'openai_api_key'


def _set_api_key():
    print('Reading OpenAI credentials')
    with open(OPENAI_CREDENTIAL_FILE, 'r') as f:
        credential = json.load(f)

    os.environ["OPENAI_API_KEY"] = credential[KEY_OPENAI_API]


class LangChainTextGenerator:
    def __init__(self, temperature=0.3):
        # load openai credentials
        _set_api_key()

        print('Starting LangChainTextGenerator...')

        self.llm = OpenAI(temperature=temperature)

    def generate_response(self, prompt, input):
        language_prompt = PromptTemplate(
            input_variables=["input"],
            template=prompt,
        )

        # print(language_prompt, language_prompt.format(input=input))
        return self.llm(language_prompt.format(input=input)).strip()

    def generate_response(self, prompt):
        return self.llm(prompt).strip()
