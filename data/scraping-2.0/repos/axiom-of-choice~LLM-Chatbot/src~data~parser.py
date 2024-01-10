from docx import Document
import os
from tqdm.auto import tqdm

from uuid import uuid4
from typing import List, Dict, Tuple, Optional, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter


from langchain.schema import Document
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings.base import Embeddings

import tiktoken

import pinecone

from config import *

# Recursive searching files
import glob

# Click oprtions and env options
import click
from dotenv import find_dotenv, load_dotenv

# Format PDF and JSON
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders.csv_loader import CSVLoader
import pandas as pd

# Utils and logging
from src.utils import connect_index
import logging
from settings import logging_config
import logging.config

logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)

load_dotenv()


def getText_docx(file: Document) -> str:
    content = []
    for paragraph in file.paragraphs:
        print(paragraph.text)
        content.append(paragraph.text)
    return "\n".join(content)


def writeText(content: str, filename: str, base_path: Optional[str] = DATA_DIR):
    write_dir = os.path.join(base_path, "raw", filename)
    with open(write_dir, "w") as f:
        f.write(content)
    return f"File {filename} written in {write_dir}"


# create the length function
def tiktoken_len(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20,
    # length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)

text_splitter_tokens = TokenTextSplitter(
    chunk_size=1000, chunk_overlap=10, encoding_name="cl100k_base"
)  # This the encoding for text-embedding-ada-002


def loadFilesinDirectory(path: str, glob: Optional[str] = None) -> List[Document]:
    if glob is None:
        loader = DirectoryLoader(path=path)
        logger.info(f"Loading files from {path} withhout glob")
    else:
        loader = DirectoryLoader(
            path=path,
            loader_kwargs={"glob": glob},
            use_multithreading=True,
            show_progress=True,
        )
        logger.info(f"Loading files from {path} with glob {glob}")
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} files")
    return docs


def loadPDFs(path: str, glob: Optional[str] = "*.pdf") -> List[Document]:
    if glob is None:
        loader = PyPDFDirectoryLoader(path=path, recursive=True)
        logger.info(f"Loading PDFs from {path} without glob")
    else:
        loader = PyPDFDirectoryLoader(path=path, glob=glob, recursive=True)
        logger.info(f"Loading PDFs from {path} with glob {glob}")
    # docs = []
    # for file in os.listdir(path):
    #    if file.endswith(".pdf"):
    #        print(os.path.join(path, file))
    #        loader = PyPDFLoader(os.path.join(path, file))
    #        pages = loader.load_and_split()
    #        docs.extend(pages)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} PDFs")
    return docs


def load_JSONL(path: str) -> List[Document]:
    docs = []
    files = glob.iglob(str(path) + "/**/*.json", recursive=True)
    for file in files:
        print(os.path.join(path, file))
        loader = JSONLoader(os.path.join(path, file), jq_schema=".flavor_text_entries[].flavor_text")
        pages = loader.load()
        docs.extend(pages)
    logger.info(f"Loaded {len(docs)} JSON")
    return docs


def load_CSV(path: str) -> List[Document]:
    docs = []
    files = glob.iglob(str(path) + "/**/*.csv", recursive=True)
    for file in files:
        pth = os.path.join(path, file)
        print(pth)
        header = list(pd.read_csv(pth, nrows=1))
        loader = CSVLoader(pth, csv_args={"fieldnames": header})
        pages = loader.load()
        docs.extend(pages)
    logger.info(f"Loaded {len(docs)} CSV")
    return docs


def embed_documents_batch(docs: List[Document]) -> List[Document]:
    embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model=COHERE_MODEL_NAME)
    embeded_docs = []
    logger.info(f"Embedding {len(docs)} documents")
    for doc in tqdm(docs):
        embeded_docs.append(embeddings.embed_documents(doc.page_content))
    return embeded_docs


def insert_embedded_documents(
    documents: List[Document],
    embeddings: Embeddings,
    index: pinecone.Index,
    batch_limit: int = 100,
    data_source: str = "Local",
    **metadata_dict: Optional[Dict[str, Any]],
) -> None:
    batch_limit = 100

    texts = []
    metadatas = []

    record_texts = documents
    logger.info(f"Embedding {len(record_texts)} documents")
    for i, record in enumerate(tqdm(documents)):
        # first get metadata fields for this record
        source = record.metadata["source"].split("/")[-1]
        page = str(record.metadata.get("page"))
        if len(metadata_dict) > 0:
            metadata = metadata_dict
        else:
            metadata = {
                "id": uuid4().hex,
                "source": source,
                "page": page,
                "data_source": data_source,
            }
        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record.page_content)
        # create individual metadata dicts for each chunk
        record_metadatas = [{"chunk": j, "text": text, **metadata} for j, text in enumerate(record_texts)]
        # # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
        # if we have reached the batch_limit we can add texts
        if len(texts) >= batch_limit:
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embeddings.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []

    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embeddings.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))


def parse(
    input_filepath: str,
    output_filepath: str,
    index_name: str,
    embeddings_model_name: str,
    glob: str = GLOB,
    data_source: str = "Local",
):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("Making final data set from raw data")
    logger.info(f"Using {embeddings_model_name} to embed documents")
    logger.info(f"Using {index_name} to connect to Pinecone index")
    documents = loadFilesinDirectory(path=input_filepath, glob=glob)
    print(len(documents))
    documents.extend(loadPDFs(path=input_filepath))
    documents.extend(load_JSONL(path=input_filepath))
    documents.extend(load_CSV(path=input_filepath))
    embeddings = AVAILABLE_EMBEDDINGS[embeddings_model_name]
    index = connect_index(index_name)
    insert_embedded_documents(documents=documents, embeddings=embeddings, index=index, data_source=data_source)


@click.command()
@click.option("--input_filepath", type=click.Path(exists=True), default=DATA_DIR)
@click.option("--output_filepath", type=click.Path(), default=DATA_DIR)
@click.option("--index_name", type=str, default=PINECONE_INDEX_NAME)
@click.option("--embeddings_model_name", type=str, default="Cohere")
@click.option("--glob", type=str, default=None)
def main(
    input_filepath: str,
    output_filepath: str,
    index_name: str,
    embeddings_model_name: str,
    glob: str = GLOB,
):
    parse(
        input_filepath=input_filepath,
        output_filepath=output_filepath,
        index_name=index_name,
        embeddings_model_name=embeddings_model_name,
        glob=glob,
    )


if __name__ == "__main__":
    logger.info("Starting embedding process")
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main(
        # input_filepath=DATA_DIR,
        # output_filepath=DATA_DIR,
        # index_name=PINECONE_INDEX_NAME,
        # embeddings_model_name="Cohere",
        # glob=GLOB,
    )
