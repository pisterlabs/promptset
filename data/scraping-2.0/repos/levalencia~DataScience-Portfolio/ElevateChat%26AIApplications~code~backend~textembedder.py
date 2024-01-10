import openai
import os
import re
import logging


class TextEmbedder():
    """
    A class for generating embeddings for text using the OpenAI API.
    """

    # Set OpenAI API credentials and deployment
    openai.api_type = "azure"
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = f"https://{os.getenv('AZURE_OPENAI_SERVICE_NAME')}.openai.azure.com/"
    openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    def clean_text(self, text, text_limit=7000):
        """
        Cleans up text by removing line breaks and truncating it if necessary.

        Args:
            text (str): The text to clean up.
            text_limit (int): The maximum length of the text.

        Returns:
            str: The cleaned up text.
        """
        # Clean up text (e.g. line breaks, )
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[\n\r]+', ' ', text).strip()
        # Truncate text if necessary (e.g. for, ada-002, 4095 tokens ~ 7000 chracters)
        if len(text) > text_limit:
            logging.warning("Token limit reached exceeded maximum length, truncating...")
            text = text[:text_limit]
        return text

    def generate_embeddings(self, text, clean_text=True):
        """
        Generates embeddings for the given text using the OpenAI API.

        Args:
            text (str): The text to generate embeddings for.
            clean_text (bool): Whether to clean up the text before generating embeddings.

        Returns:
            list: A list of embeddings for the text.
        """
        if clean_text:
            text = self.clean_text(text)
        response = openai.Embedding.create(input=text, engine=self.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        embeddings = response['data'][0]['embedding']
        return embeddings
