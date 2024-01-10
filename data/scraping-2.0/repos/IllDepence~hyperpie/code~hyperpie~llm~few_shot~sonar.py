import importlib.util
from typing import Any, Dict, List

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, Extra, root_validator


class SonarEmbeddings(BaseModel, Embeddings):
    """Embeddings by SONAR models.

    It only supports the 'text_sonar_basic_encoder' model.
    See: https://github.com/facebookresearch/SONAR/blob/main/sonar/store/cards/text_sonar_basic_encoder.yaml

    Attributes:
        client (Any): TextToEmbeddingModelPipeline.
        source_lang (str): Source language code.

    Methods:
        embed_documents(texts: List[str]) -> List[List[float]]:
            Generates embeddings for a list of documents.
        embed_query(text: str) -> List[float]:
            Generates an embedding for a single piece of text.
    """

    client: Any  # The Spacy model loaded into memory

    source_lang: str

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid  # Forbid extra attributes during model initialization

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validates that the SONAR package is available.

        Args:
            values (Dict): The values provided to the class constructor.

        Returns:
            The validated values.

        Raises:
            ValueError: If the SONAR package is not available.
        """
        if values["source_lang"] is None:
            values["source_lang"] = "eng_Latn"

        # Check if the Spacy package is installed
        if importlib.util.find_spec("sonar") is None:
            raise ValueError(
                "SONAR package not found. Please install it. "
                "(See https://github.com/facebookresearch/SONAR)"
            )
        try:
            # Try to load the 'en_core_web_sm' Spacy model
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

            values["client"] = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder"
            )
        except OSError:
            # If the model is not found, raise a ValueError
            raise ValueError(
                "SONAR model 'text_sonar_basic_encoder' not found. "
            )
        return values  # Return the validated values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of documents.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Returns:
            A list of embeddings, one for each document.
        """
        # torch.tensor in shape (len(texts), 1024)
        embeddings = self.client.predict(texts, source_lang=self.source_lang)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Generates an embedding for a single piece of text.

        Args:
            text (str): The text to generate an embedding for.

        Returns:
            The embedding for the text.
        """
        embeddings = self.client.predict([text], source_lang=self.source_lang)
        return embeddings[0].tolist()

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously generates embeddings for a list of documents.
        This method is not implemented and raises a NotImplementedError.

        Args:
            texts (List[str]): The documents to generate embeddings for.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Asynchronous embedding generation is not supported.")

    async def aembed_query(self, text: str) -> List[float]:
        """
        Asynchronously generates an embedding for a single piece of text.
        This method is not implemented and raises a NotImplementedError.

        Args:
            text (str): The text to generate an embedding for.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError("Asynchronous embedding generation is not supported.")
