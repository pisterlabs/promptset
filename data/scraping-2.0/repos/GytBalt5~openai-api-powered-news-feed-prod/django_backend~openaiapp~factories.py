from abc import ABC, abstractmethod
from typing import List, Union

from django.conf import settings

import openai
from pandas import DataFrame
from scrapy.spiders import CrawlSpider

from openaiapp.spiders import NewsSpider
from openaiapp.tokenizers import AbstractTokenizer, Tokenizer
from openaiapp.embeddings import AbstractEmbeddings, TextEmbeddings, DataFrameEmbeddings
from openaiapp.text_preparators import (
    AbstractTextPreparatory,
    TextPreparatory,
    DataFrameTextPreparatory,
)
from openaiapp.ai_question_answering import (
    AbstractAIQuestionAnswering,
    AIQuestionAnsweringBasedOnContext,
)


# Set the OpenAI API key from Django settings.
openai.api_key = settings.OPENAI_API_KEY


class Factory(ABC):
    """
    Abstract base class for factories that create various objects.
    """

    @abstractmethod
    def create_object(self, *args, **kwargs):
        """
        Abstract method to create an object.
        """
        pass


class SpiderFactory(Factory):
    """
    Factory for creating web spider objects.
    """

    def create_object(self, domain: str, start_urls: List[str]) -> CrawlSpider:
        """
        Create a NewsSpider object for web crawling.

        :param domain: The domain for the spider.
        :param start_urls: A list of URLs where the spider starts crawling.
        :return: An instance of NewsSpider.
        """
        return NewsSpider(domain=domain, start_urls=start_urls)


class TokenizerFactory(Factory):
    """
    Factory for creating tokenizer objects.
    """

    TOKENIZER_ENCODING = "cl100k_base"

    def create_object(self, encoding: str = TOKENIZER_ENCODING) -> AbstractTokenizer:
        """
        Create a Tokenizer object with the specified encoding.

        :param encoding: The encoding to be used by the tokenizer.
        :return: An instance of Tokenizer.
        """
        return Tokenizer(encoding=encoding)


class EmbeddingsFactory(Factory):
    """
    Factory for creating embeddings objects.
    """

    EMBEDDING_ENGINE = "text-embedding-ada-002"

    def create_object(
        self,
        input_type: Union[str, DataFrame],
        embedding_engine: str = EMBEDDING_ENGINE,
    ) -> AbstractEmbeddings:
        """
        Create an embeddings object based on the input type.

        :param input_type: A string or DataFrame for which embeddings are to be created.
        :param embedding_engine: The engine to use for creating embeddings.
        :return: An instance of AbstractEmbeddings.
        :raises TypeError: If the input type is not supported.
        """
        if input_type == str:
            return TextEmbeddings(embedding_engine=embedding_engine)
        elif input_type == DataFrame:
            return DataFrameEmbeddings(embedding_engine=embedding_engine)
        else:
            raise TypeError(f"Unsupported input type: {input_type}.")


class TextPreparatoryFactory(Factory):
    """
    Factory for creating text preparatory objects.
    """

    MIN_TOKENS = 8
    MAX_TOKENS = 512

    def create_object(self, df: DataFrame = None) -> AbstractTextPreparatory:
        """
        Create a text preparatory object, optionally based on a DataFrame.

        :param df: An optional DataFrame for text preparation.
        :return: An instance of AbstractTextPreparatory.
        """
        tokenizer = TokenizerFactory().create_object()
        if df is None:
            return TextPreparatory(tokenizer=tokenizer)
        else:
            return DataFrameTextPreparatory(
                df=df,
                tokenizer=tokenizer,
                min_tokens=self.MIN_TOKENS,
                max_tokens=self.MAX_TOKENS,
            )


class AIQuestionAnsweringFactory(Factory):
    """
    Factory for creating AI question answering objects.
    """

    MODEL = "gpt-3.5-turbo-instruct"
    ANSWER_MAX_TOKENS = 256
    CONTEXT_MAX_LEN = 2048

    def create_object(
        self,
        text_embeddings_object: AbstractEmbeddings,
        text_preparatory: AbstractTextPreparatory,
        stop_sequence: str = None,
        model: str = MODEL,
        answer_max_tokens: int = ANSWER_MAX_TOKENS,
        context_max_len: int = CONTEXT_MAX_LEN,
    ) -> AbstractAIQuestionAnswering:
        """
        Create an AIQuestionAnsweringBasedOnContext object.

        :param text_embeddings_object: An AbstractEmbeddings object for text embedding.
        :param text_preparatory: An AbstractTextPreparatory object for text preparation.
        :param stop_sequence: A sequence indicating where to stop the answer generation.
        :param model: The model to use for question answering.
        :param answer_max_tokens: The maximum number of tokens for the answer.
        :param context_max_len: The maximum length of the context.
        :return: An instance of AIQuestionAnsweringBasedOnContext.
        """
        return AIQuestionAnsweringBasedOnContext(
            text_preparatory=text_preparatory,
            text_embeddings_object=text_embeddings_object,
            model=model,
            max_tokens=answer_max_tokens,
            context_max_len=context_max_len,
            stop_sequence=stop_sequence,
        )


class OpenAIAppObjectFactory(Factory):
    """
    Abstract factory for creating various OpenAI app-related objects.
    Note: This class is for learning purposes.
    """

    def __init__(self, factory: Factory):
        self.factory = factory

    def create_object(self, *args, **kwargs):
        """
        Create an object using the specified factory.

        :return: An object created by the specified factory.
        """
        return self.factory.create_object(*args, **kwargs)
