from typing import List

import numpy as np
from pydantic import BaseModel

from langchain.embeddings.base import Embeddings

from LibertyAI.liberty_config import get_configuration
import requests

class LibertyEmbeddings(Embeddings, BaseModel):

    endpoint: str

    def _get_embedding(self, text: str) -> List[float]:
        config = get_configuration()
        json_data = {
            'text' : text,
            #'API_KEY': config.get('API', 'KEY'),
        }
        response = requests.post(
            self.endpoint,
            json = json_data,
        )

        try:
            reply = response.json()['embedding']
        except:
            reply = []

        return reply

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._get_embedding(text)
