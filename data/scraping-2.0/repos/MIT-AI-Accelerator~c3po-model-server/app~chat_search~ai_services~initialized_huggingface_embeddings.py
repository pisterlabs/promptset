from typing import AnyStr, Any
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

class InitializedHuggingFaceEmbeddings(HuggingFaceEmbeddings):

    loaded_model: Any

    def __init__(self, **kwargs: AnyStr):
        """Initialize the sentence_transformer."""
        try:
            super().__init__(**kwargs)
        except Exception as e:
            self.client = self.loaded_model