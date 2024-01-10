import pandas as pd
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

import sys
import os
sys.path.append("../")
from src.config import Configuration


class HybridRag:


    def __init__(self, data_path) -> None:
        self.chunk_size = 1600
        self.chunk_overlap = 200
        self.model = "sentence-transformers/LaBSE"
        self.path = data_path
    

    def _build(self):
        file = os.path.join(self.path, "test_major/docs.csv")
        docs = pd.read_csv(file)
        docs = docs[docs['id'] >= 6]
        db = []

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n","\n\n", "\n"],
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        for i, data in docs.iterrows():
            doc = text_splitter.create_documents([data['content']])
            [d.metadata.update({"id": data['id']}) for d in doc]
            db.extend(doc)

        conf = Configuration()

        self.bm25_retriever = BM25Retriever.from_documents(db)
        self.bm25_retriever.k = 2
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key="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",
            model_name=self.model
            )
        self.elastic_vector_search = ElasticsearchStore(
            es_connection=conf.load_elasticsearch_connection(),
            index_name="test-basic",
            embedding=embeddings,
            distance_strategy="EUCLIDEAN_DISTANCE")

        es_retriever = self.elastic_vector_search.as_retriever(search_kwargs={"k": 2})

        # initialize the ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, es_retriever], weights=[0.5, 0.5]
        )
        return
    
    def get_retriever(self):
        return self.ensemble_retriever
