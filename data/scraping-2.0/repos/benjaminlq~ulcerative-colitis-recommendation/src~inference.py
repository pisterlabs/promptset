"""QA Retrieval Modules
"""
import random
import re
import time
from typing import Callable, Dict, Literal, Optional

import openai
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.vectorstores import FAISS

from config import MAX_TOKENS, TEMPERATURE, TOP_P


class Retrieval_Interface:
    """Interface for QA Retrieval Module"""

    def __init__(
        self,
        llm_type: str,
        emb_path: str,
        openai_api_key: str,
        embedding_type: str = "text-embedding-ada-002",
        embedding_store: Literal["faiss"] = "faiss",
    ):
        """Interface for QA Retrieval Module

        Args:
            llm_type (str): Type of LLM model to be used for extraction
            emb_path (str): Path to embedding datastore.
            openai_api_key (str): OpenAI API Key
            embedding_type (str, optional): Type of Embedding model to be used.
                Convert the query into embedding and find the most relevant text chunk from document
                store based on similarity (e.g. cosine distance). Defaults to "text-embedding-ada-002".
            embedding_store (Literal[faiss], optional): Type of document store to use. Defaults to "faiss".
        """
        self.prompt_template = None
        self.llm_type = llm_type
        self.chain = None
        if embedding_store.lower() == "faiss":
            self.embedder = OpenAIEmbeddings(
                model=embedding_type, openai_api_key=openai_api_key
            )
            self.docsearch = FAISS.load_local(emb_path, self.embedder)
            print("FAISS Datastore successfully loaded")
        else:
            raise Exception("Please specify correct embedding store type.")
        self.token_counter = 0
        self.dollar_counter = 0.0

    def __call__(self, query: str) -> str:
        """Response to a query

        Args:
            query (str): User query

        Returns:
            str: Output response
        """
        result = self._response(query)
        response = Retrieval_Interface._format_result(query, result)
        return response

    def _retry_with_exponential_backoff(
        func: Callable,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 5,
        errors: tuple = (openai.error.RateLimitError,),
    ):
        """Retry a function with exponential backoff

        Args:
            func (Callable): Function to wrap around.
            initial_delay (float, optional): Initial Delay. Defaults to 1.
            exponential_base (float, optional): Exponential Base. Defaults to 2.
            jitter (bool, optional): Jitter. Defaults to True.
            max_retries (int, optional): max_retries. Defaults to 5.
            errors (tuple, optional): Errors. Defaults to (openai.error.RateLimitError,).
        """

        def wrapper(*args, **kwargs):
            """Wrapper Function
            Returns:
                Callable: function output
            """
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            # Loop until a successful response or max_retries is hit or an exception is raised
            while True:
                try:
                    return func(*args, **kwargs)

                # Retry on specified errors
                except errors:
                    # Increment retries
                    num_retries += 1

                    # Check if max retries has been reached
                    if num_retries > max_retries:
                        raise Exception(
                            f"Maximum number of retries ({max_retries}) exceeded."
                        )

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Sleep for the delay
                    time.sleep(delay)

                # Raise exceptions for any errors not specified
                except Exception as e:
                    raise e

        return wrapper

    @_retry_with_exponential_backoff
    def _response(self, query: str) -> Dict:
        """Call QA Chain on input query

        Args:
            query (str): User Query

        Returns:
            Dict: Output Dictionary with fields "query", "answer", "source_documents"
        """
        return self.chain(query)

    @classmethod
    def _get_chain(
        cls,
        llm: Callable,
        docsearch: Callable,
        prompt_template: BasePromptTemplate,
        verbose: Optional[bool] = None,
    ):
        """Generate the QA Chain

        Args:
            llm (Callable): LLM to use for extracting information from text chunks
            docsearch (Callable): Retrieval Object to search for relevant text chunks
                based on embeddings similarity.
            prompt_template (BasePromptTemplate): Prompt Template
            verbose (Optional[bool]): Whether chain query returns prompt and COT from LLM. Default to None.

        Returns:
            RetrievalQAWithSourcesChain: Chain Object Instance
        """
        return RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=docsearch.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            verbose=verbose,
            chain_type_kwargs={"prompt": prompt_template},
            reduce_k_below_max_tokens=True,
        )

    @classmethod
    def _format_result(cls, query: str, result: Dict) -> str:
        """Format Output Dict into String Response
        Args:
            query (str): Input query
            result (Dict): Output dictionary

        Returns:
            str: AI Assistant Response
        """
        sources = [
            str(re.findall(r"[ \w-]+?(?=\.)", name)[0])
            for name in (
                list(
                    set([doc.metadata["source"] for doc in result["source_documents"]])
                )
            )
        ]
        response = f"""### Scenario:
        {query}
        \n### Response:
        {result['answer']}
        \n### Relevant sources:
        {', '.join(sources)}
        """
        return response


class ChatOpenAIRetrieval(Retrieval_Interface):
    """ChatOpenAIRetrieval Module"""

    def __init__(
        self,
        prompt_template: ChatPromptTemplate,
        emb_path: Callable,
        openai_api_key: str,
        llm_type: str = "gpt-3.5-turbo",
        embedding_type: str = "text-embedding-ada-002",
        embedding_store: Literal["faiss"] = "faiss",
        temperature: float = TEMPERATURE,
        top_p: float = TOP_P,
        max_tokens: int = MAX_TOKENS,
        verbose: Optional[bool] = None,
    ):
        """AI QARetrievalFromSource Module

        Args:
            system_template (str): System Template sets the context and expected behaviour of the LLM.
            user_template (str): Specific User Input which requires user to enter a query for RetrievalQA
                chain to extract relevant information related to sources.
            emb_path (Callable): Path to embedding datastore.
            openai_api_key (str): OpenAI API Key
            llm_type (str, optional): Type of LLM model to be used for extraction. Defaults to "gpt-3.5-turbo".
            embedding_type (str, optional): Type of Embedding model to be used. Defaults to "text-embedding-ada-002".
            embedding_store (Literal[faiss], optional): Type of document store to use. Defaults to "faiss".
            temperature (float, optional): Temperature to use in Softmax function during decoding.
                0 means deterministic and higher (max=2) means more random. Defaults to config.TEMPERATURE.
            top_p (float, optional): Pick the minimum number of tokens with highest probability with cumulative
                probability greater than top_p. Between 0 and 1. Defaults to config.TOP_P.
            max_tokens (int, optional): Maximum number of output tokens to ADD to input tokens.
                Defaults to config.MAX_TOKENS.
            verbose (Optional[bool]): Whether chain query returns prompt and COT from LLM. Default to None.
        """
        super(ChatOpenAIRetrieval, self).__init__(
            llm_type, emb_path, openai_api_key, embedding_type, embedding_store
        )
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.llm_type = llm_type
        self.llm = ChatOpenAI(
            model_name=self.llm_type,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
        )
        self.prompt_template = prompt_template
        self.chain = ChatOpenAIRetrieval._get_chain(
            llm=self.llm,
            docsearch=self.docsearch,
            prompt_template=self.prompt_template,
            verbose=verbose,
        )
