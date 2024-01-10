import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import click
import chromadb
import torch
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModel
import os

# from dotenv import load_dotenv
from chromadb.config import Settings
from langchain.document_loaders import CSVLoader, PDFMinerLoader, TextLoader, UnstructuredExcelLoader, Docx2txtLoader, JSONLoader
from vtt_loader import VttLoader
from better_text_loader import BetterTextLoader

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME = "custom_data"

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"
INGEST_THREADS = os.cpu_count() or 8
DOCUMENT_MAP = {
    ".md": BetterTextLoader,
    ".py": TextLoader,
    ".pdf": PDFMinerLoader,
    ".csv": CSVLoader,
    ".xls": UnstructuredExcelLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".vtt": VttLoader,
    ".txt": BetterTextLoader,
}

class DocumentLoader:
    def __init__(self, source_dir):
        self.source_dir = source_dir

    def load_single_document(self, file_path: str) -> Document:
        compatible_loaders = [key for key in DOCUMENT_MAP.keys() if file_path.endswith(key)]
        file_extension = max(compatible_loaders, key=len)
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            loader = loader_class(file_path)
        else:
            raise ValueError("Document type is undefined")
        return loader.load()[0]

    def load_documents(self) -> list[Document]:
        all_files = os.listdir(self.source_dir)
        paths = [os.path.join(self.source_dir, file_path) for file_path in all_files if any(file_path.endswith(ext) for ext in DOCUMENT_MAP.keys())]
        n_workers = min(INGEST_THREADS, max(len(paths), 1))
        docs = []
        with ProcessPoolExecutor(n_workers) as executor:
            futures = [executor.submit(self.load_single_document, path) for path in paths]
            for future in as_completed(futures):
                docs.append(future.result())
        return docs

class DocumentProcessor:
    def __init__(self, device_type):
        self.embedding_function = HuggingFaceBgeEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": device_type},
            encode_kwargs = {'normalize_embeddings': True}
        )

        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT, settings=Settings(allow_reset=True))

        # TODO: provide option to reset collection
        # chroma_client.reset()
        chroma_client.create_collection(CHROMA_COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        self.langchain_chroma = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_function,
        )

    def process_documents(self, directories):
        for dir in directories:
            logging.info(f"Loading documents from {dir}")

            loader = DocumentLoader(dir)
            documents = loader.load_documents()

            for doc in documents:
                custom_docs = self.text_splitter.split_documents([doc])
                self.langchain_chroma.add_documents(custom_docs)

            logging.info(f"Loaded {len(documents)} documents from {dir}. Split into {len(custom_docs)} chunks of text.")

@click.command()
@click.option("--source_dirs", help="Directories containing documents to ingest", multiple=True)
@click.option( "--device_type", default="cuda" if torch.cuda.is_available() else "cpu", type=click.Choice(["cpu","cuda"]), help="Device to run on. (Default is cuda)")
def main(device_type, source_dirs):
    processor = DocumentProcessor(device_type)
    processor.process_documents(source_dirs)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()

