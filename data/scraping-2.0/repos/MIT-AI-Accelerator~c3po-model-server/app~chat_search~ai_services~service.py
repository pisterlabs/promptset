import re
from typing import Any
from .initialized_huggingface_embeddings import InitializedHuggingFaceEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from pydantic import BaseModel

from app.aimodels.gpt4all.ai_services.completion_inference import (
    CompletionInference,
    CompletionInferenceInputs,
)
from app.chat_search.ai_services.marco_rerank_retriever import MarcoRerankRetriever
from app.core.errors import ValidationError
from app.core.minio import download_pickled_object_from_minio
from app.core.model_cache import MODEL_CACHE_BASEDIR
from app.core.config import settings

from sqlalchemy.orm import Session
from minio import Minio
from app.aimodels.bertopic.crud import (
    bertopic_embedding_pretrained as bertopic_embedding_pretrained_crud,
    document as document_crud,
)

from sample_data import CHAT_DATASET_1_PATH


class RetrievalService(BaseModel):
    completion_inference: CompletionInference
    db: Session | None = None
    s3: Minio | None = None
    sentence_model: Any | None = None  #: :meta private:

    class Config:
        arbitrary_types_allowed = True

    def retrieve(self, api_inputs: CompletionInferenceInputs, summarize=False, max_docs=1000):
        # validate input
        if not isinstance(api_inputs, CompletionInferenceInputs):
            raise ValidationError("must input type CompletionInferenceInputs")

        retriever = self._build_retriever(channel_names=[CHAT_DATASET_1_PATH], max_docs=max_docs)

        if summarize:
            llm = self.completion_inference._build_llm(api_inputs)
            results = self._retrieve_and_summarize(
                llm,
                query=api_inputs.prompt,
                retriever=retriever,
            )
        else:
            results = self._retrieve_only(
                query=api_inputs.prompt,
                retriever=retriever,
            )

        return results

    def _build_retriever(
        self,
        channel_names=[],
        max_docs=1000,
    ):
        if self.sentence_model is None:
            try:
                sentence_model_db_obj = (
                    bertopic_embedding_pretrained_crud.get_by_model_name(
                        db=self.db, model_name="all-MiniLM-L6-v2"
                    )
                )

                self.sentence_model = download_pickled_object_from_minio(
                    id=sentence_model_db_obj.id, s3=self.s3
                )

                local_embeddings = InitializedHuggingFaceEmbeddings(
                    loaded_model=self.sentence_model
                )
            except Exception as _:
                # failed to load from db or minio, so load from huggingface if possible

                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                local_embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # get DocumentModel objects from the database
        document_models = document_crud.get_by_created_date_range(
            db=self.db, start_date=None, end_date=None, limit=max_docs
        )

        # convert DocumentModel objects to Document objects
        documents = []
        for doc_model in document_models:
            source_link = (
                ""
                if not doc_model.mattermost_document
                or len(doc_model.mattermost_document) == 0
                else f"{settings.mm_aoc_base_url}/pl/{doc_model.mattermost_document[0].message_id}"
            )

            document = Document(
                page_content=doc_model.text,
                metadata={
                    "originated_from": doc_model.originated_from,
                    "original_created_time": doc_model.original_created_time,
                    "link": source_link,
                },
            )

            documents.append(document)

        chat_retriever = FAISS.from_documents(
            documents, local_embeddings
        ).as_retriever()
        chat_retriever.search_kwargs = {"k": 25}

        def bm25_preprocess_func(text):
            # replace non alphanumeric characters with whitespace
            text = re.sub(r"[^a-zA-Z0-9]", " ", text)

            # lowercase and split on whitespace
            return text.lower().split()

        # initialize the bm25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents, preprocess_func=bm25_preprocess_func
        )
        bm25_retriever.k = 25

        # initialize the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chat_retriever], weights=[0.5, 0.5]
        )

        # download marco rerank model
        marco_pretrained_obj = bertopic_embedding_pretrained_crud.get_by_model_name(
            db=self.db, model_name="ms-marco-TinyBERT-L-6"
        )

        cross_model = download_pickled_object_from_minio(
            id=marco_pretrained_obj.id, s3=self.s3
        )

        # initialize the rerank retriever
        rerank_retriever = MarcoRerankRetriever(
            base_retriever=ensemble_retriever,
            cross_model=cross_model,
            rerank_model_name_or_path="cross-encoder/ms-marco-TinyBERT-L-6",
            max_relevant_documents=20,
        )

        return rerank_retriever

    def _retrieve_only(self, query=None, retriever=None):
        result = {"input": query, "result": "No LLM used to summarize"}
        result["source_documents"] = retriever.get_relevant_documents(query)
        return result

    def _retrieve_and_summarize(self, llm, query=None, retriever=None):
        ###Unknown: how to address FAISS chunking and add metadata
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="input",
            return_source_documents=True,
            verbose=True,
        )

        result = chain({"input": f"{query}"})
        return result
