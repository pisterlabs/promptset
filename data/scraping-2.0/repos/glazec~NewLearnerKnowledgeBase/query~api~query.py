from http.server import BaseHTTPRequestHandler, HTTPServer

from urllib.parse import urlparse, parse_qs
import cohere
import pinecone
import numpy as np
import json
import os
import jieba
import chinese_converter
from os.path import join

co = cohere.Client(os.environ["COHERE_KEY"])


def single_word_search(texts, query):
    index = np.char.count(texts, query)
    filtered_arr = texts[index != 0]
    if len(filtered_arr) < 5:
        return filtered_arr
    sorted_indices = np.argsort(index)
    # take the top 5 results
    # top_five_counts = index[sorted_indices[-5:]]
    top_five_texts = texts[sorted_indices[-5:]]

    return top_five_texts


def full_text_search(query):
    with open(join("data", "texts.txt"), "r") as file:
        data = file.read()
    texts = np.array([text.strip() for text in data.split("===")])
    query = query.lower()
    query = chinese_converter.to_simplified(query)
    tokens = jieba.cut(query)
    stopwords = [
        line.strip()
        for line in open(
            join("data", "stop_words.txt"), "r", encoding="utf-8"
        ).readlines()
    ]
    filtered_words = [word for word in tokens if word not in stopwords]
    response = np.array([])
    for word in filtered_words:
        response = np.concatenate((response, single_word_search(texts, word)), axis=0)
    response = np.unique(response)
    print(len(response))
    return response


def semantic_search(query):
    # get cohere key from environment variable
    pinecone.init(
        api_key=os.environ["PINECONE_KEY"],
        environment="gcp-starter",
    )

    index_name = "cohere-newlearner"
    index = pinecone.Index(index_name)
    # create the query embedding
    xq = co.embed(
        texts=[query],
        model="embed-multilingual-v2.0",
        input_type="search_query",
        truncate="END",
    ).embeddings

    # query, returning the top 5 most similar results
    res = index.query(xq, top_k=5, include_metadata=True)
    return res.to_dict()

    # for match in res["matches"]:
    #     print(f"{match['score']:.2f}: {match['metadata']['text']}")


def rerank_object_to_json(rerank_object):
    result = []
    for i in rerank_object:
        result.append({"text": i.document["text"], "score": i.relevance_score})
    return result


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urlparse(self.path).query
        query_params = parse_qs(query)
        print(query_params)
        # Now query_params is a dictionary of your query parameters
        semantic_res = semantic_search(query_params["query"][0])
        semantic_res = np.array(
            [i["metadata"]["text"] for i in semantic_res["matches"]]
        )
        full_text_res = full_text_search(query_params["query"][0])
        rerank_candidate = np.concatenate((semantic_res, full_text_res), axis=0)
        rerank_candidate = np.unique(rerank_candidate)
        print(rerank_candidate)
        rerank_result = co.rerank(
            query=query_params["query"][0],
            documents=rerank_candidate,
            top_n=5,
            model="rerank-multilingual-v2.0",
        )

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps({"matches": rerank_object_to_json(rerank_result)}).encode()
        )
        return

    # def do_POST(self):
    #     content_length = int(self.headers["Content-Length"])  # Gets the size of data
    #     post_data = self.rfile.read(content_length)  # Gets the data itself
    #     data = json.loads(
    #         post_data.decode("utf-8")
    #     )  # Parses it from JSON to a Python dictionary

    #     res = semantic_search(data["query"])

    #     self.send_response(200)
    #     self.send_header("Content-type", "application/json")
    #     self.end_headers()
    #     self.wfile.write(json.dumps(res).encode())
    #     return


def run(server_class=HTTPServer, handler_class=handler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting httpd on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
    print(full_text_search("我们Macos的快捷键是什么"))
