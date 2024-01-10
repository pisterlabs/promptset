# -*- coding: utf-8 -*-
# @Team: AIBeing
# @Author: huaxinrui@tal.com
# pip install chromadb-client
import chromadb
import hashlib
from typing import  List
from core.log import logger
from interact.llm.vector.base import Vector
from langchain.embeddings import OpenAIEmbeddings
from core.conf import config
hash = hashlib.md5()

try:
    db = chromadb.HttpClient(host=config.vector_host, port=str(config.vector_port))
except:
    raise RuntimeError("chromadb not connect %s:%s" % (config.vector_host, config.vector_port))

assert config.llm_embedding_type in ["openai", "msai"], "llm_type must be openai or flag"

if config.llm_embedding_type == "msai":
    try:
        from FlagEmbedding import FlagModel
        model = FlagModel(config.llm_embedding if config.llm_embedding else 'BAAI/bge-large-zh')
    except:
        logger.error("model not found")

class VectorDB(Vector):
    def __init__(self, typ: str = "openai"):
        self.typ = typ
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if self.typ == "openai":
            return OpenAIEmbeddings().embed_documents(texts)
        res =  model.encode(texts)
        return res.tolist()
    async def async_get_embeddings(self, texts: List[str]) -> List[float]:
        if self.typ == "openai":
            res = await OpenAIEmbeddings().aembed_documents(texts)
            return res[0]
        res = model.encode(texts)
        return res.tolist()

    def md5(self, text: str) -> str:
        hash.update(text.encode())
        return hash.hexdigest()

    def add(self, collection: str, docs: List[str]):
        collection = db.get_collection(name=collection)
        ids = [self.md5(i) for i in docs]
        collection.add(documents=docs, ids=ids, embeddings=self.get_embeddings(docs))

    def similarity(self, collection: str, doc: str, tok_k: int) -> List[str]:
        collection = db.get_collection(name=collection)
        res = collection.query(
            query_embeddings=self.get_embeddings([doc]),
            n_results=tok_k,
            # include=["documents"]
        )
        # distance: [[0.08322394639891817, 0.17011645508128126]]
        return res['documents'][0]

    async def async_similarity(self, collection: str, doc: str, tok_k: int) -> List[str]:
        collection = db.get_collection(name=collection)
        res = collection.query(
            query_embeddings=await self.async_get_embeddings([doc]),
            n_results=tok_k,
            # include=["documents"]
        )
        return res['documents'][0]

    def count(self, collection: str) -> int:
        collection = db.get_collection(name=collection)
        return collection.count()

    def delete_collection(self, collection: str):
        try:
            db.delete_collection(name=collection)
        except:
            logger.error("collection %s not exist" % (collection))

    def create_collection(self, collection: str):
        """raise Exception if exist"""
        try:
            db.create_collection(name=collection, metadata={"hnsw:space": "cosine"})
        except:
            logger.error("collection %s already exist" % (collection))


if __name__ == "__main__":
    collection = "mixiaoquan"
    vdb = VectorDB(config.llm_embedding_type)
    vdb.delete_collection(collection)  #
    vdb.create_collection(collection)  #
    docs = ["我们一起来玩吧", "天青色等烟雨", "这雨下的可真大啊", "你知道我叫什么吗", "我是华心瑞"]
    vdb.add(collection, docs)
    print(vdb.count(collection))
    res = vdb.similarity(collection, "我叫什么?", 2)
    print(res)
