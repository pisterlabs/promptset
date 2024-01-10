"""
File that contains the logic for text segmentation and summarization.
"""
import logging
from typing import List

import tiktoken

from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources.loading import BaseCombineDocumentsChain

import src.logic.config.secrets as config_secrets

class TextSummarizer():
    """
    Class that contains the logic for text segmentation and summarization.
    """

    def __init__(self) -> None:
        self.chain: BaseCombineDocumentsChain = load_summarize_chain(
            llm = ChatOpenAI(
                temperature = 0.5,
                client = Document,
                openai_api_key = config_secrets.read_openai_credentials(),
            ),
            chain_type = "map_reduce",
            verbose = True,
        )
    
    def num_tokens(self, text: str, model: str = "cl100k_base") -> int:
        """
        Calculate the number of tokens in a given text.

        :param content: The string to be tokenized.
        :param model: The model to be used for tokenization.
        :return: The total number of tokens in the text.
        :raise ValueError: If the param text is not a string or if the string is empty.
        :raise ValueError: If the param model is not a string or if the string is empty.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Argument text must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Argument model must be a non-empty string.")
        try:
           tokens: List[int] = tiktoken.get_encoding(model).encode(text)
        except ValueError as e:
            raise ValueError(
                f"Error encoding document with model '{model}': {str(e)}"
            ) from e
        return len(tokens)

    def summarize_text(
        self, raw_text: str, max_tokens: int, model: str = "cl100k_base"
    ) -> str:
        """
        Calulate the number of tokens in a given text and summarizes it if it has more
        than a specified amount of tokens.

        :param raw_text: The text to be processed.
        :param max_tokens: The maximum number of tokens allowed for a prompt to the
        respective LLM.
        :return: The summarized text if the original document has more than the specified
        amount of tokens, otherwise the original document.
        :raise ValueError: If arg raw_text is not a string or if the string is empty.
        :raise ValueError: If arg max_tokens is not a positive integer.
        """
        if not isinstance(raw_text, str) or not raw_text:
            raise ValueError("Argument text must be a non empty string")
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Argument max_tokens must be a positive integer.")
        try:
            if self.num_tokens(text = raw_text, model = model) > max_tokens:
                documents: Document = [Document(page_content = raw_text)]
                chunk_size: int = max_tokens // 2
                chunk_overlap: int = max_tokens // 10
                contents: List[Document] = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size, chunk_overlap = chunk_overlap
                ).split_documents(documents = documents)
                analysis_result: str = self.chain.run(contents)
                return analysis_result
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {str(e)} ") from e
        return raw_text
