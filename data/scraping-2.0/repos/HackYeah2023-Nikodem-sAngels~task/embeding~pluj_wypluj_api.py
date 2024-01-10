import pandas as pd
import pickle
import psycopg2

import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import openai as openai
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

EMBEDDING_MODEL = "text-embedding-ada-002"
with open(".env") as f:
    key = f.read().split("=")[1].replace("\n", "")
    openai.api_key = key

FILE = "tmp"

embedding_cache_path = f"cache/{FILE}.pkl"

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)


def embedding_from_string(
    string: str, model: str = EMBEDDING_MODEL, embedding_cache=embedding_cache
) -> list:
    """Return embedding of given string, using a cache to avoid recomputing."""
    if (string, model) not in embedding_cache.keys():
        embedding_cache[(string, model)] = get_embedding(string, model)
        with open(embedding_cache_path, "wb") as embedding_cache_file:
            pickle.dump(embedding_cache, embedding_cache_file)
    return embedding_cache[(string, model)]


class SimpleRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)

        self.send_header("Content-type", "text/plain")
        self.end_headers()

        content_length = int(self.headers["Content-Length"])

        post_data = self.rfile.read(content_length)

        vec = embedding_from_string(post_data.decode("utf-8"))

        self.wfile.write(str(vec).encode("utf-8"))


def run(server_class=HTTPServer, handler_class=SimpleRequestHandler, port=3333):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)

    print(f"Starting server on port {port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
