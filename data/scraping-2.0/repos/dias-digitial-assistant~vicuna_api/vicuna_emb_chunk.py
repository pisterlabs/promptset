from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    Union,
)
import numpy as np
from pydantic import BaseModel
from langchain.embeddings.base import Embeddings
import  requests
from dotenv import load_dotenv
load_dotenv()
import os
import time
from tenacity import retry, stop_after_attempt, wait_fixed
class VicunaEmbeddings(BaseModel, Embeddings):
    uid: str = None
    server_url:str = None
    client: Any  #: :meta private:
    model: str = "vicuna-13b"
    embedding_ctx_length: int = 2000
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Tuple[()]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""

    @retry(stop=stop_after_attempt(6), wait=wait_fixed(2))
    def _embed(self, texts: Union[List[str],str]):
        headers = {"Content-Type": "application/json"}
        if isinstance(texts, str):
            texts = [texts]
        
        response = requests.post(
            self.server_url+"/worker_get_embeddings",
            json={
                "input": texts,
                "model": self.model
            },
            headers=headers,
        )
        try:
            if response.status_code == 200:
                json_data = response.json()
                return json_data["embedding"]   
        except:
            if response.status_code != 200:
                print(response.content)
            else:
                print("Data formatting incorrect, received server code", response.status_code, response.content, response.content==None)

    def _get_len_safe_embeddings(self, texts: List[str], chunk_size=None)-> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]

        try:
            
            from transformers import AutoTokenizer
            
            tokens = []
            indices = []

            tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF")

            for i, text in enumerate(texts):
                text = text.replace("\n"," ")
                token = tokenizer.encode(text)
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens+=[token[j : j + self.embedding_ctx_length]]
                    indices+=[i]
            
            batched_embeddings = []
            _chunk_size = chunk_size or self.chunk_size
            for i in range(0,len(tokens)):
                for j in range(0,len(tokens[i]),_chunk_size):
                    print(i,j)
                    decoded_tokens = [tokenizer.decode(tokens[i][j:j+_chunk_size], skip_special_tokens=True)]
                    response = self._embed(texts=decoded_tokens)
                    batched_embeddings +=[r for r in response]

            results: List[List[List[float]]] = [[] for _ in range(len(texts))]
            lens: List[List[int]] = [[] for _ in range(len(texts))]
            for i in range(len(indices)):
                results[indices[i]].append(batched_embeddings[i])
                lens[indices[i]].append(len(batched_embeddings[i]))

            for i in range(len(texts)):
                average = np.average(results[i], axis=0, weights=lens[i])
                embeddings[i] = (average / np.linalg.norm(average)).tolist()

            return embeddings

        except:
            raise ValueError(
                "Could not embed"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to Hosted OpenAIs Document endpoint.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        return self._get_len_safe_embeddings(texts)
        

    def embed_query(self, text: str) -> List[float]:
        """Call out to  Hosted OpenAIs, query embedding endpoint
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        emb = self._get_len_safe_embeddings([text])
        return emb[0]
