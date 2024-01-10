import re
from threading import Lock
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import FAISS
from pydantic import BaseModel, validator
from aimbase.services import (
    SentenceTransformerInferenceService,
    CrossEncoderInferenceService,
)
from .marco_rerank_retriever import MarcoRerankRetriever
from .stis_embeddings import STISEmbeddings
from .document_crud import CRUDDocument
from sqlalchemy.orm import Session
from minio import Minio

# TODO: ensure all services are Session - safe with db (this one is)
# TODO: ensure all long-lasting services are thread-safe for model inference (this one is)
class RetrievalService(BaseModel):
    # must be provided and initialized (see validators)
    sentence_inference_service: SentenceTransformerInferenceService
    cross_encoder_inference_service: CrossEncoderInferenceService

    # must be provided
    document_crud: CRUDDocument

    # optional, must be provided if minio is used
    s3: Minio | None = None

    # internal only
    _retriever_lock: Lock = Lock() # :meta private:

    @validator("sentence_inference_service", pre=True, always=True)
    def sentence_service_must_be_initialized(
        cls, v: SentenceTransformerInferenceService
    ) -> SentenceTransformerInferenceService:
        if not v.initialized:
            raise ValueError(
                "sentence_inference_service not initialized.  Please call initialize() on the SentenceTransformerInferenceService first."
            )
        return v

    @validator("cross_encoder_inference_service", pre=True, always=True)
    def cross_encoder_service_must_be_initialized(
        cls, v: CrossEncoderInferenceService
    ) -> CrossEncoderInferenceService:
        if not v.initialized:
            raise ValueError(
                "cross_encoder_inference_service not initialized.  Please call initialize() on the CrossEncoderInferenceService first."
            )
        return v

    class Config:
        arbitrary_types_allowed = True

    def retrieve(self, db: Session, query: str):
        if not isinstance(query, str):
            raise ValueError("must input query as str")

        retriever = self._build_retriever(db)

        if retriever is None:
            return {"input": query, "result": "No documents in database"}

        results = self._get_documents(
            query=query,
            retriever=retriever,
        )

        return results

    def _build_retriever(self, db: Session):
        with self._retriever_lock: # lock to ensure only one thread at a time can access this method
            local_embeddings = STISEmbeddings(
                sentence_inference_service=self.sentence_inference_service
            )

            # get DocumentModel objects from the database
            document_models = self.document_crud.get_by_created_date_range(
                db=db, start_date=None, end_date=None
            )

            # convert DocumentModel objects to Document objects
            documents = []
            for doc_model in document_models:
                document = Document(
                    page_content=doc_model.text,
                    metadata={
                        "id": doc_model.id,
                        "original_created_time": doc_model.original_created_time,
                        "source": doc_model.source,
                    },
                )

                # TODO: add in hook here for adjusting document metadata

                documents.append(document)

            if len(documents) == 0:
                return None

            # TODO: add in capability to save and load FAISS index from minio
            # adjust the FAISS index in a different area using endpoint on doc save
            # TODO: build poetry groups & docs to use gpu or cpu version of faiss
            document_retriever = FAISS.from_documents(
                documents, local_embeddings
            ).as_retriever()
            document_retriever.search_kwargs = {"k": 100}

            def bm25_preprocess_func(text):
                # replace non alphanumeric characters with whitespace
                text = re.sub(r"[^a-zA-Z0-9]", " ", text)

                # lowercase and split on whitespace
                return text.lower().split()

            # initialize the bm25 retriever
            bm25_retriever = BM25Retriever.from_documents(
                documents, preprocess_func=bm25_preprocess_func
            )
            bm25_retriever.k = 100

            # initialize the ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, document_retriever], weights=[0.5, 0.5]
            )

            rerank_retriever = MarcoRerankRetriever(
                base_retriever=ensemble_retriever,
                cross_model=self.cross_encoder_inference_service.model,
                rerank_model_name_or_path="cross-encoder/ms-marco-TinyBERT-L-6",
                max_relevant_documents=200,
            )

            return rerank_retriever

    def _get_documents(self, query=None, retriever=None):
        result = {"input": query, "result": "No LLM used to summarize"}
        result["source_documents"] = retriever.get_relevant_documents(query)
        return result