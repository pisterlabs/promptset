from elasticsearch import Elasticsearch
from langchain.schema.document import Document
from typing import List

class MyElasticsearch(Elasticsearch):
    def __init__(self, url, index_name="linewell-policy") -> None:
        self.index_name = index_name
        self.client = Elasticsearch(url)
        self.fields = ["标题", "子标题", "内容"]

    def search(self, query, top_k=0, index_name=None, fields=["*"]) -> List[Document]:
        index_name = self.index_name if index_name is None else index_name
        query_body = {
            "query": {
                "multi_match": {
                    "analyzer": "ik_smart",
                    "query": query,
                    "type": "cross_fields", # for structured data
                    "fields": fields,
                }
            }    
        }
        response = self.client.search(index=index_name, body=query_body)
        docs = [
            Document(
                page_content="\n".join([hit["_source"][field] for field in self.fields]),
                metadata={
                    "score": hit["_score"],
                    "source": hit['_source']['标题'],
                }
            ) if index_name is None else \
            Document(
                page_content="\n".join([f"{k}: {v}" for k, v in hit["_source"].items()]),
                metadata={
                    "score": hit["_score"],
                    "source": hit['_source']['项目名称'],
                }
            ) if index_name == "project" else \
            Document(
                page_content=hit["_source"]["content"],
                metadata={
                    "score": hit["_score"],
                    "source": hit['_source']['title'],
                }
            )
            for hit in response["hits"]["hits"]
        ]
        top_k = len(docs) if top_k <= 0 else top_k
        return docs[:top_k]
