"""
Langchain LLM pipeline for generative QA pipeline.
"""


from dataclasses import dataclass, field, InitVar
from haystack.document_stores import FAISSDocumentStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import logging
from pathlib import Path
import sys
from typing import Iterable, Literal, Optional, Union


from docs2chat.config import Config, config
from docs2chat.preprocessing.utils import (
    create_vectorstore,
    langchain_to_haystack_docs,
    load_and_split_from_dir,
    load_and_split_from_str,
    _EmbeddingsProtocol,
    _RetrieverProtocol,
    _TextSplitterProtocol
)


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s: %(message)s"
)
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@dataclass
class ExtractivePreProcessor:

    LOADER_FACTORY = {
        "text": load_and_split_from_str,
        "dir": load_and_split_from_dir
    }
    
    content: Union[str, list[str]] = field(default=config.DOCUMENTS_DIR)
    docs: Optional[list] = field(default=None)
    load_from_type: str = field(default="dir")
    text_splitter: Optional[_TextSplitterProtocol] = field(default=None)

    def __post_init__(self):
        if self.load_from_type not in ["text", "dir"]:
            raise ValueError(
                "`load_from_type` must be one of `text` or `dir`."
            )
        if self.text_splitter is None:
            _logger.info(
                "Generating text splitter."
            )
            text_splitter = CharacterTextSplitter(        
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            setattr(self, "text_splitter", text_splitter)
    
    def load_and_split(self, show_progress=True, store=False):
        load_func = ExtractivePreProcessor.LOADER_FACTORY[self.load_from_type]
        docs = langchain_to_haystack_docs(load_func(
            content=self.content,
            text_splitter=self.text_splitter,
            show_progress=show_progress
        ))
        if store:
            setattr(self, "docs", docs)
        return docs
    
    def create_vectorstore(self, store=False):
        vectorstore = FAISSDocumentStore(
            sql_url="sqlite:///",
            embedding_dim=384
        )
        if store:
            setattr(self, "vectorstore", vectorstore)
        return vectorstore

    def preprocess(
        self,
        show_progress: bool = True,
        return_vectorstore: bool = True,
        store_docs: bool = False,
        store_vectorstore: bool = True
    ):
        _logger.info(
            "Loading documents into vectorstore. "
            "This may take a few mquitinutes ..."
        )
        docs = self.load_and_split(
            show_progress=show_progress,
            store=store_docs
        )
        vectorstore = self.create_vectorstore(
            store=store_vectorstore
        )
        vectorstore.write_documents(docs)
        if store_vectorstore:
            setattr(self, "vectorstore", vectorstore)
        if return_vectorstore:
            return vectorstore
        return


@dataclass
class GenerativePreProcessor:

    LOADER_FACTORY = {
        "text": load_and_split_from_str,
        "dir": load_and_split_from_dir
    }
    
    content: Union[str, list[str]] = field(default=config.DOCUMENTS_DIR)
    docs: Optional[list] = field(default=None)
    embeddings: Optional[_EmbeddingsProtocol] = field(default=None)
    load_from_type: str = field(default="dir")
    text_splitter: Optional[_TextSplitterProtocol] = field(default=None)
    
    def __post_init__(self):
        if self.load_from_type not in ["text", "dir"]:
            raise ValueError(
                "`load_from_type` must be one of `text` or `dir`."
            )
        if self.text_splitter is None:
            _logger.info(
                "Generating text splitter."
            )
            text_splitter = CharacterTextSplitter(        
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            setattr(self, "text_splitter", text_splitter)
        if self.embeddings is None:
            _logger.info(
                f"Loading embedding model from {config.EMBEDDING_DIR}."
            )
            embeddings = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_DIR
            )
            setattr(self, "embeddings", embeddings)
    
    def load_and_split(self, show_progress=True, store=False):
        load_func = GenerativePreProcessor.LOADER_FACTORY[self.load_from_type]
        docs = load_func(
            content=self.content,
            text_splitter=self.text_splitter,
            show_progress=show_progress
        )
        if store:
            setattr(self, "docs", docs)
        return docs
    
    def create_vectorstore(self, docs, store=False):
        vectorstore = create_vectorstore(
            docs=docs,
            embeddings=self.embeddings
        )
        if store:
            setattr(self, "vectorstore", vectorstore)
        return vectorstore

    def preprocess(
        self,
        show_progress: bool = True,
        return_vectorstore: bool = True,
        store_docs: bool = False,
        store_vectorstore: bool = False
    ):
        _logger.info(
            "Loading documents into vectorstore. This may take a few minutes ..."
        )
        docs = self.load_and_split(
            show_progress=show_progress,
            store=store_docs
        )
        vectorstore = self.create_vectorstore(
            docs=docs,
            store=store_vectorstore
        )
        if return_vectorstore:
            return vectorstore
        return


class PreProcessor:

    preprocessor_dict = {
        "search": ExtractivePreProcessor,
        "snip": ExtractivePreProcessor,
        "generative": GenerativePreProcessor
    }

    def __new__(
        cls,
        chain_type: Literal["search", "snip", "generative"],
        **kwargs
    ):
        preprocessor_cls = cls.preprocessor_dict[chain_type]
        return preprocessor_cls(**kwargs)
