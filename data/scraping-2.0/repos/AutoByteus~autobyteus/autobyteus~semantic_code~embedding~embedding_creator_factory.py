"""
embedding_creator_factory.py

This module contains the get_embedding_creator function which is responsible for getting instances of embedding
creator classes based on a given configuration. It supports getting different types of embedding
creators, such as OpenAIEmbeddingCreator and SentenceTransformerEmbeddingCreator. Each type of embedding creator 
is implemented as a singleton, ensuring only one instance of each type can exist.
"""

from autobyteus.semantic_code.embedding.base_embedding_creator import BaseEmbeddingCreator
from autobyteus.semantic_code.embedding.openai_embedding_creator import OpenAIEmbeddingCreator
from autobyteus.semantic_code.embedding.sentence_transformer_embedding_creator import SentenceTransformerEmbeddingCreator
from autobyteus.config import config


def get_embedding_creator() -> BaseEmbeddingCreator:
    """
    Gets an instance of an embedding creator class based on the configuration. 
    If the instance does not exist, it is created due to the singleton nature of the classes. 

    Returns:
    An instance of an embedding creator class.
    """
    embedding_type = config.get('DEFAULT_EMBEDDING_TYPE', 'sentence_transformer')
    if embedding_type == 'openai':
        return OpenAIEmbeddingCreator()
    elif embedding_type == 'sentence_transformer':
        return SentenceTransformerEmbeddingCreator()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
