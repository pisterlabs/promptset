# -*- coding: utf-8 -*-
import os
from typing import Dict, List
from loguru import logger

from elasticsearch import Elasticsearch
from tqdm import tqdm
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from langchain.vectorstores import FAISS
from .process_data import Embeddings, load_document

ES_STORE_ROOT_PATH = os.getenv("ES_STORE_ROOT_PATH")
ES_URL = os.getenv("ES_URL")


def _default_knn_mapping(dims: int) -> Dict:
    """Generates a default index mapping for kNN search."""
    return {
        "properties": {
            "text": {"type": "text"},
            "vector": {
                "type": "dense_vector",
                "dims": dims,
                "index": True,
                "similarity": "cosine",
            },
        }
    }


def generate_search_query(vec, size) -> Dict:
    query = {
        "query": {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                    "params": {
                        "queryVector": vec
                    }
                }
            }
        },
        "size": size
    }
    return query


def generate_knn_query(vec, size) -> Dict:
    query = {
        "knn": {
            "field": "vector",
            "query_vector": vec,
            "k": 10,
            "num_candidates": 100
        },
        "size": size
    }
    return query


def generate_hybrid_query(text, vec, size, knn_boost) -> Dict:
    query = {
        "query": {
            "match": {
                "text": {
                    "query": text,
                    "boost": 1 - knn_boost
                }
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": vec,
            "k": 10,
            "num_candidates": 100,
            "boost": knn_boost
        },
        "size": size
    }
    return query


class ElasticsearchServer:
    def __init__(self):
        self.client = Elasticsearch(
            ES_URL,
            verify_certs=False,
        )
        self.embedding = Embeddings()
        self.es = ElasticsearchStore(
            index_name='audit_index',
            embedding=self.embedding,
            es_connection=self.client,
        )

    def create_index(self, index_name: str):
        if not self.client.indices.exists(index=index_name):
            dims = len(self.embedding.embed_query("test"))
            mapping = _default_knn_mapping(dims)
            self.client.indices.create(index=index_name, body={"mappings": mapping})
            logger.info(f"Successfully Created Index: {index_name}!")
        else:
            logger.info(f"Index: {index_name} already exists!")

    def doc_upload(self, index_name: str, data_url: str):
        self.create_index(index_name)

        docs = []
        for root, dirs, files in os.walk(data_url):
            for file in tqdm(files):
                file_path = os.path.join(root, file)
                res = load_document(file_path)
                if res:
                    self.es.add_documents(res)
                    logger.info(f"Successfully inserted document {res[0].metadata}!")
        logger.info("Successfully inserted documents!")

    def doc_search(
            self, method: str, query: str, top_k: int, knn_boost: float, index_name: str
    ) -> List[Dict]:
        result = []
        query_vector = self.embedding.embed_query(query)
        if method == "knn":
            query_body = generate_knn_query(vec=query_vector, size=top_k)
        elif method == "hybrid":
            query_body = generate_hybrid_query(text=query, vec=query_vector, size=top_k, knn_boost=knn_boost)
        else:
            query_body = generate_search_query(vec=query_vector, size=top_k)

        response = self.client.search(index=index_name, body=query_body)
        hits = [hit for hit in response["hits"]["hits"]]
        for i in hits:
            result.append(
                {
                    "content": i["_source"]["text"],
                    'source': i["_source"]["metadata"]["source"],
                    'score': i["_score"]
                }
            )
        return result

    def delete(self, index_name):
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"Successfully Deleted Index: {index_name}!")


if __name__ == '__main__':
    # 创建一个ElasticsearchServer实例
    index_name = 'audit_index'

    es_server = ElasticsearchServer()
    # es_server.delete(index_name)
    data_url = os.path.join(ES_STORE_ROOT_PATH, '231118')
    es_server.doc_upload(index_name=index_name, data_url=data_url)
    # a = es_server.doc_search(index_name=index_name, query='如何认定本罪的标准', top_k=5, method='knn', knn_boost=0.5)
    # print(a)
