import os

import openai
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]


class AudioTranscript:
    def __init__(self, loader):
        self.loader = loader
        self._docs = None

    @property
    def docs(self):
        return self._docs

    def execute(self):
        parser = OpenAIWhisperParser()

        loader = GenericLoader(self.loader, parser)
        try:
            self._docs = loader.load()
        except Exception as e:
            print(f"Error: {e}")
        return self
