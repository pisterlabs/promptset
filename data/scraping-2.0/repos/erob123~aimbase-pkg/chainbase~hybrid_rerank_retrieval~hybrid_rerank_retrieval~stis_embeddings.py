from typing import Any
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from aimbase.services import SentenceTransformerInferenceService
from pydantic import validator


class STISEmbeddings(HuggingFaceEmbeddings):
    """
    Sentence Transformer Inference Service Embeddings

    Same params as HuggingFaceEmbeddings from langchain, with an additional
    SentenceTransformerInferenceService object to allow loading from
    minio without instantiating another model in memory.
    """

    sentence_inference_service: SentenceTransformerInferenceService
    """The SentenceTransformerInferenceService containing the embedding model."""

    @validator("sentence_inference_service", pre=True, always=True)
    def model_must_be_initialized(
        cls, v: SentenceTransformerInferenceService
    ) -> SentenceTransformerInferenceService:
        if not v.initialized:
            raise ValueError(
                "sentence_inference_service not initialized.  Please call initialize() on the SentenceTransformerInferenceService first."
            )
        return v

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer client for the langchain class."""
        super(HuggingFaceEmbeddings, self).__init__(**kwargs)

        # don't need to check for sentence transformers package, that is done inside SentenceTransformerInferenceService
        self.client = self.sentence_inference_service.model
