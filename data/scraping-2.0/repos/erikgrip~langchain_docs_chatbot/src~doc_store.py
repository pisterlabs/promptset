# pylint: disable=unspecified-encoding
import logging
import os
import shutil
from glob import glob

from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from tqdm import tqdm

from src.utils.log import logger


load_dotenv()
logging.getLogger("unstructured").setLevel(logging.WARNING)

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 100
FILE_EXTS = (".md", ".mdx")


class DocStore:
    """Document store using Chroma."""

    def __init__(
        self,
        data_path,
        persist_dir="data/chroma",
        delete_persisted_db=False,
    ):
        """Initialize DocStore."""
        self.data_path = data_path
        self.persist_dir = persist_dir

        self.doc_loader = UnstructuredMarkdownLoader
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        self.embedding = OpenAIEmbeddings()
        if delete_persisted_db:
            if os.path.exists(self.persist_dir):
                logger.info("Removing persisted data...")
                shutil.rmtree(self.persist_dir)

        logger.info("Initializing Chroma database...")
        self.db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding,
        )
        if self.is_empty():
            logger.info("No documents found in database. Creating a new one...")
            self._db_from_docs_dir(self.data_path)

    def is_empty(self):
        """Check if the database is empty."""
        return len(self.db.get(include=[])["ids"]) == 0

    def size(self):
        """Return the size of the database."""
        return len(self.db.get(include=[])["ids"])

    def persist(self):
        """Persist the database."""
        logger.info("Writing Chroma database to disk (%s)...", self.persist_dir)
        self.db.persist()

    def as_retriever(self, num_retrieved_docs):
        """Return the database as a retriever."""
        return self.db.as_retriever(search_kwargs={"k": num_retrieved_docs})

    def add_docs(self, docs, max_retries=4):
        """Add documents to Chroma database with retry."""
        retries = 0
        try:
            self.db.add_documents(docs)
        except KeyError:
            retries += 1
            if retries > max_retries:
                raise
            logger.error("Failed to add documents to Chroma database. Retrying...")
            self.add_docs(docs)

    def _db_from_docs_dir(self, dir_path):
        """Load all documents from a directory and its subdirectories."""
        docs = self._load_docs_from_dir(dir_path)
        split_docs = self.text_splitter.split_documents(docs)

        logger.info("Loading Chroma database from %s...", dir_path)
        chunk_size = 50
        for i in tqdm(range(0, len(split_docs), chunk_size)):
            chunk_end = min(i + chunk_size, len(split_docs))
            self.add_docs(split_docs[i:chunk_end])
        logger.info("Done!")

    def _load_docs_from_dir(self, dir_path):
        """Load all documents from a directory and its subdirectories."""
        loaders = [
            self.doc_loader(path)
            for path in glob(dir_path + "/**", recursive=True)
            if path.endswith(FILE_EXTS)
        ]
        logging.info("Loading documents...")
        docs = []
        for loader in tqdm(loaders):
            docs.extend(loader.load())
        return docs
