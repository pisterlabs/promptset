import os
import argparse
from flask import Flask, request, jsonify
from flask_caching import Cache
from sentence_transformers import SentenceTransformer
import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(os.getcwd(), "log.txt"))
    ]
)

app = Flask(__name__)


@app.route('/embed_query', methods=['POST'])
def embed_query():
    sentence = request.json['sentence']
    key = f"{args.model_name}_query_{sentence}"

    # Check if the embeddings are in the cache
    embeddings = cache.get(key)
    if embeddings is None:
        # Compute the embeddings and store them in the cache
        embeddings = model.encode(sentence, normalize_embeddings=True).tolist()
        cache.set(key, embeddings)

    return jsonify(embeddings)


@app.route('/embed_documents', methods=['POST'])
def embed_documents():
    sentences = request.json['sentences']
    key = f"{args.model_name}_documents_{'_'.join(sentences)}"

    # Check if the embeddings are in the cache
    embeddings = cache.get(key)
    if embeddings is None:
        # Compute the embeddings and store them in the cache
        embeddings = model.encode(sentences, normalize_embeddings=True).tolist()
        cache.set(key, embeddings)

    return jsonify(embeddings)

import requests
from typing import List
from abc import ABC, abstractmethod
from langchain.embeddings.base import Embeddings
import torch

EMPTY_STRING = "EMPTY DOCUMENT STRING PLACEHOLDER"

class EmbeddingClient(Embeddings):
    def __init__(self, server_url):
        self.server_url = server_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # PATCH CODE for some models which have dimensionaility errors for empty strings
        texts = [text if text else EMPTY_STRING for text in texts]

        response = requests.post(f"{self.server_url}/embed_documents", json={'sentences': texts})
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        # PATCH CODE for some models which have dimensionaility errors for empty strings
        text = text if text else EMPTY_STRING
        response = requests.post(f"{self.server_url}/embed_query", json={'sentence': text})
        return response.json()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-small-en')
    parser.add_argument('--port', type=int, default=8002)
    parser.add_argument('--folder', type=str, required=True)
    args = parser.parse_args()

    # Initialize the model
    model = SentenceTransformer(args.model_name, device=args.device).to(torch.device(args.device))

    # Initialize the cache
    os.makedirs(os.path.join(os.getcwd(), args.folder), exist_ok=True)
    cache_dir = os.path.join(os.getcwd(), args.folder, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = Cache(app, config={'CACHE_TYPE': 'filesystem', 'CACHE_DIR': cache_dir,
                               'CACHE_DEFAULT_TIMEOUT': 7 * 24 * 60 * 60})

    app.run(port=args.port, threaded=True, processes=1)
