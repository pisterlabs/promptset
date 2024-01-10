import redis
import os
from dotenv import load_dotenv
from langchain.schema import Document
from typing import List, Dict
import json
from langchain.vectorstores.redis import Redis
import sys
sys.path.append(f"../")
from botcore.setup import get_openai_embeddings, load_my_env

class RedisDB:

    def __init__(self):

        load_my_env()
        self.embeddings = get_openai_embeddings()
        self.url = os.getenv("REDIS_CLOUD")
        self.limit = 0.2

    def json_to_doc(self, data: Dict, meta_info: Dict = None) -> Document:
        """
            data = {"title": str, "features": [], "post_id": str, ...}
        """
        feats = ", ".join([i for i in data['features']])
        txt = f"{data['title']}. {feats}"
        return Document(page_content=txt, metadata=meta_info)

    def add_doc(self, doc: Document, index_name: str):
        try:
            
            Redis.from_documents([doc], self.embeddings, redis_url=self.url, index_name=index_name)

            return True
        except:
            print("An exception occurred when adding new doc")
            return False
    ## add
    def add_new_wanted(self, data: Dict):
        p = data["product"].replace(" ","_")
        index = f'wanted:{p}'
        doc = self.json_to_doc(data, {"type": index})
        return self.add_doc(doc, index)

    def add_new_stock(self, data: Dict):
        p = data['product'].replace(" ","_")
        index = f"stock:{p}"
        doc = self.json_to_doc(data, {"type": index})
        return self.add_doc(doc, index)

    def search_in_wanted(self, data: Dict):
        p = data["product"].replace(" ","_")
        index_name = f"wanted:{p}"
        return self.search_doc(data, index_name)

    
    def search_doc(self, data: Dict, index: str):
        redis = Redis(redis_url = self.url, index_name = index,\
                    embedding_function=self.embeddings.embed_query)
        doc = self.json_to_doc(data, {"type": index})
        query = doc.page_content
        try:
            results = redis.similarity_search_limit_score(query, score_threshold=self.limit)
            return results
        except:
            return False

        
