#!/usr/bin/env python3
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import os
import glob
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders import AsyncHtmlLoader

# from llama_hub.web.async_web.base import AsyncWebPageReader
# for jupyter notebooks uncomment the following two lines of code:
# import nest_asyncio
# nest_asyncio.apply()

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.document_transformers import Html2TextTransformer

# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


class DocsStore:
    """

    """
    def __init__(self, config: Dict[str, Any],
                 **kwargs) -> None:
        #  Load environment variables
        self.config: Dict[str, Any] = config
        self.model_name: str = config["model_name"]
        self.openai_api_key: str = config["OPENAI_API_KEY"]

        if self.openai_api_key is None or self.openai_api_key == "":
            print("OPENAI_API_KEY is not set")

            # exit(1)
        # else:
        #    print(
        #        f"OPENAI_API_KEY is set: {self.openai_api_key[0:3]}...{self.openai_api_key[-4:]}")

        self.persist_directory: str = config['PERSIST_DIRECTORY']  # 'db'
        self.source_directory: str = config['SOURCE_DIRECTORY']  # "data/raw"
        self.embeddings_model_name: str = config['EMBEDDINGS_MODEL_NAME']
        self.chunk_size: int = int(config['CHUNK_SIZE'])
        self.chunk_overlap: int = int(config['CHUNK_OVERLAP'])

        self.embeddings = None
        self.splitter = None
        self.llm = None
        self.db = None
        self.retriever = None
        self.qa = None

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

        # print(f"documents from dir: {self.source_directory}")
        # print(f"db dir: {self.persist_directory}")
        return

    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        ext = "." + file_path.rsplit(".", 1)[-1]
        if ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[ext]
            loader = loader_class(file_path, **loader_args)
            return loader.load()

        raise ValueError(f"Unsupported file extension '{ext}'")

    def load_documents(self, ignored_files: List[str] = [], progress: bool = False) -> List[Document]:
        """
        Loads all documents from the source documents directory, ignoring specified files
        """
        all_files: list = []
        for ext in LOADER_MAPPING:
            all_files.extend(
                glob.glob(os.path.join(self.source_directory,
                        f"**/*{ext}"), recursive=True)
            )
        filtered_files = [
            file_path for file_path in all_files if file_path not in ignored_files]

        results: list = []
        if progress:
            with Pool(processes=os.cpu_count()) as pool:
                with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
                    for i, docs in enumerate(pool.imap_unordered(DocsStore.load_document, filtered_files)):
                        results.extend(docs)
                        pbar.update()
        else:
            for doc in filtered_files:
                docs = DocsStore.load_document(doc)
                results.extend(docs)

        return results

    def process_documents(self, ignored_files: List[str] = [], verbose: bool = False) -> List[Document]:
        """
        Load documents and split in chunks
        """
        if verbose:
            print(f"Loading documents from {self.source_directory}")
        documents = self.load_documents(ignored_files)
        if not documents:
            if verbose:
                print("No new documents to load")
            # exit(0)
            return None
        if verbose:
            print(
                f"Loaded {len(documents)} new documents from {self.source_directory}")
        texts = self.splitter.split_documents(documents)
        if verbose:
            print(
                f"Split into {len(texts)} chunks of text (max. {self.chunk_size} tokens each)")
        return texts

    def vectorstore_exist(self) -> bool:
        """
        Checks if vectorstore exists
        """
        return False

    @staticmethod
    def load_urls(urls: List[str] = [], to_text: bool = True) -> List[Document]:
        if len(urls) == 0:
            return []

        loader = AsyncHtmlLoader(urls)
        docs = loader.load()

        # loader = AsyncWebPageReader()
        # docs = loader.load_data(urls)

        if to_text:
            html2text = Html2TextTransformer()
            docs_transformed = html2text.transform_documents(docs)
            return docs_transformed

        return docs
