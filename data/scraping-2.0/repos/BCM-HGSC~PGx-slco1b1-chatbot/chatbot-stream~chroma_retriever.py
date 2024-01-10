# -*- coding:utf-8 -*-
# Created by liwenw at 9/11/23

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document

from langchain.vectorstores import Chroma

from omegaconf import OmegaConf
from chromadb.config import Settings

class ChromaRetriever:
    def __init__(self, config):
        self.config = config

    def get_vector_store(self):
        embeddings = OpenAIEmbeddings()
        collection_name = self.config.chromadb.collection_name
        persist_directory = self.config.chromadb.persist_directory
        chroma_db_impl = self.config.chromadb.chroma_db_impl
        vector_store = Chroma(collection_name=collection_name,
                              embedding_function=embeddings,
                              client_settings=Settings(
                                  chroma_db_impl=chroma_db_impl,
                                  persist_directory=persist_directory
                              ),
                              )
        return vector_store

    def max_marginal_relevance_search(self, query):
        vector_store = self.get_vector_store()
        return vector_store.max_marginal_relevance_search(query)

    def similarity_search(self, query):
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query)

    def query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        vector_store = self.get_vector_store()
        return vector_store.__query_collection(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )

def main():
    yamlfile = "/Users/liwenw/PycharmProjects/ai/PGx-slco1b1-chatbot/config-1000-50.yaml"
    config = OmegaConf.load(yamlfile)
    db = ChromaRetriever(config)
    question = "I take Crestor for my cholesterol. My husband and I are planning on a baby. Can I continue taking Crestor during my pregnancy? Will it affect my baby?"
    docs = db.max_marginal_relevance_search(question)
    print(type(docs))
    # print(docs)
    context = ""
    for doc in docs:
        context = context + doc.page_content + " "

    print(context)

if __name__ == "__main__":
    main()




