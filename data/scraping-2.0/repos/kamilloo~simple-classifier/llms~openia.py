import os
from langchain.llms import OpenAI as OpenAPI
from llms.llm import Llm
from dotenv import load_dotenv

class OpenAI(Llm):
    def __init__(self):
        load_dotenv()

        self._llm = OpenAPI(model_name='text-davinci-003')

    def get_llm(self):
        return self._llm
