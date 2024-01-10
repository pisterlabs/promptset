from pathlib import Path as P
from langchain.vectorstores import Qdrant, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredHTMLLoader
import re
from langchain.embeddings import HuggingFaceEmbeddings
import shutil
from contextlib import contextmanager
from time import perf_counter
from typing import Optional, List
import logging
from pydantic import BaseModel, Field
from langchain.document_loaders import (
    ReadTheDocsLoader,
    DirectoryLoader,
    PDFMinerLoader,
)
import tqdm
import yaml
from umbertobot.loaders import PandasLoader, get_directory_loader
from umbertobot.index_utils import update_qdrant_alias
from umbertobot.models import (
    EmbeddingConfig,
    PreprocessingConfig,
    LoaderConfig,
    PersistenceConfig,
    IndexType,
)

logging.basicConfig(level="INFO")

supported_loader_types = ["rtdocs", "html", "text_files", "csv", "parquet", "pdf"]


def load_model_from_yaml(base_model_cls, yaml_path):
    with open(yaml_path) as f:
        raw_obj = yaml.safe_load(f)
    return base_model_cls.parse_obj(raw_obj)


def load_or_get_default(base_model_cls, yaml_path):
    if yaml_path:
        return load_model_from_yaml(base_model_cls, yaml_path)
    else:
        return base_model_cls.get_default()


def strip_html_whitespaces(html_str):
    return re.sub("\n+", "\n", html_str.page_content)


def load_html_docs(path):
    html_files = list(P(path).rglob("*html"))
    docs = (UnstructuredHTMLLoader(p).load()[0] for p in html_files)
    for doc in tqdm.tqdm(docs):
        doc.page_content = re.sub("\n+", "\n", doc.page_content)
        yield doc


def load_raw_docs(loader_config: LoaderConfig):
    """
    TODO: what is actual return type? Is this list by default or
    """
    loader_type = loader_config.loader_type
    path = loader_config.path
    glob_pattern = loader_config.glob_pattern
    text_col = loader_config.text_col

    assert loader_type in supported_loader_types
    if loader_type == "rtdocs":
        return ReadTheDocsLoader(path).load()
    elif loader_type == "html":
        return load_html_docs(path)
    elif loader_type == "text_files":
        return get_directory_loader(
            path, glob=glob_pattern, gitignore_path=loader_config.gitignore_path
        ).load()
    elif loader_type in ["parquet", "csv"]:
        return PandasLoader(path, text_col, loader_type, included_cols).load()
    elif loader_type == "pdf":
        return PDFMinerLoader(path).load()


# langchain_loader = ReadTheDocsLoader("rtdocs/langchain.readthedocs.io/en/latest/")
# llama_loader = ReadTheDocsLoader("rtdocs/gpt-index.readthedocs.io/en/latest")
# llama_path = "rtdocs/gpt-index.readthedocs.io/en/latest"

# docs = llama_loader.load()


def preprocess_docs(
    raw_docs,
    preprocessing_config: PreprocessingConfig = PreprocessingConfig.get_default(),
    min_char_length=10,
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=preprocessing_config.chunk_size,
        chunk_overlap=preprocessing_config.chunk_overlap,
    )
    documents = text_splitter.split_documents(raw_docs)
    return (doc for doc in documents if len(doc.page_content) > min_char_length)


@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start
    print(f"Time: {perf_counter() - start:.3f} seconds")


class DocStoreBuilder(BaseModel):

    preprocessing_config: PreprocessingConfig
    embedding_config: EmbeddingConfig

    def make_doc_store_from_documents(
        self, documents, collection_name, persistence_config: PersistenceConfig
    ):
        logging.info(f"building a docstore using {str(persistence_config)}")
        embeddings = self.embedding_config.load_embeddings()
        persist_path = P(persistence_config.persist_directory)
        if not persist_path.exists():
            persist_path.mkdir(parents=True)
        with catchtime():
            doc_store = Chroma.from_documents(
                self.filter_texts(documents),
                embeddings,
                collection_name=persistence_config.collection_name,
                persist_directory=persistence_config.persist_directory,
            )
        return doc_store

    def _make_qdrant_doc_store(
        self, documents, embeddings, collection_name, persistence_config
    ):
        with catchtime():
            qdrant_path = (
                P(persistence_config.persist_directory)
                / "qdrant"
                / persistence_config.collection_name
            )
            if qdrant_path.exists():
                logging.info(
                    f"index persistence path exist, removing {str(qdrant_path)}"
                )
                shutil.rmtree(qdrant_path)
            qdrant_path.mkdir(parents=True)
            doc_store = Qdrant.from_documents(
                self.filter_texts(documents),
                embeddings,
                distance_func=persistence_config.distance_func,
                path=str(qdrant_path),
            )
            update_qdrant_alias(doc_store.client, collection_name)
            return doc_store

    def setup_doc_store(
        self, loader_config: LoaderConfig, persistence_config: PersistenceConfig
    ):
        raw_docs = load_raw_docs(loader_config)
        docs = preprocess_docs(raw_docs, self.preprocessing_config)
        return self.make_doc_store_from_documents(
            docs,
            collection_name=persistence_config.collection_name,
            persistence_config=persistence_config,
        )

    def get_doc_store(
        self, loader_config: LoaderConfig, persistence_config: PersistenceConfig
    ):
        # if self.check_if_exists(loader_config):
        #     return None
        # else:
        return self.setup_doc_store(loader_config, persistence_config)

    def filter_texts(self, documents):
        return [doc for doc in documents]
