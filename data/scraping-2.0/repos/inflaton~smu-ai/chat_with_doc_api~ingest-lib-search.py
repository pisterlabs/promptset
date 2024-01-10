# setting device on GPU if available, else CPU
import os
from timeit import default_timer as timer
from typing import List

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.init import *


def load_documents(source_path) -> List:
    loader = PyPDFDirectoryLoader(source_path, silent_errors=True)
    documents = loader.load()

    loader = DirectoryLoader(
        source_path, glob="**/*.html", silent_errors=True, show_progress=True
    )
    documents.extend(loader.load())
    return documents


def split_chunks(documents: List, chunk_size, chunk_overlap) -> List:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)


def generate_index(
    filename: str, chunks: List, embeddings: HuggingFaceInstructEmbeddings
) -> VectorStore:
    full_path = index_path + filename + "/"
    os.mkdir(full_path)

    if using_faiss:
        faiss_instructor_embeddings = FAISS.from_documents(
            documents=chunks, embedding=embeddings
        )

        faiss_instructor_embeddings.save_local(full_path)
        return faiss_instructor_embeddings
    else:
        chromadb_instructor_embeddings = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=full_path
        )

        chromadb_instructor_embeddings.persist()
        return chromadb_instructor_embeddings


# Constants
device_type, hf_pipeline_device_type = get_device_types()
hf_embeddings_model_name = (
    os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
)
index_path = os.environ.get("FAISS_INDEX_PATH_PDFS") or os.environ.get(
    "CHROMADB_INDEX_PATH_PDFS"
)
using_faiss = os.environ.get("FAISS_INDEX_PATH_PDFS") is not None
source_path = os.environ.get("SOURCE_PDFS_PATH")
chunk_size = os.environ.get("CHUNCK_SIZE")
chunk_overlap = os.environ.get("CHUNK_OVERLAP")

start = timer()
embeddings = HuggingFaceInstructEmbeddings(
    model_name=hf_embeddings_model_name, model_kwargs={"device": device_type}
)
end = timer()

print(f"Completed in {end - start:.3f}s")

start = timer()

if not os.path.isdir(index_path):
    print(
        f"The index persist directory {index_path} is not present. Creating a new one."
    )
    os.mkdir(index_path)

    print(f"Loading PDF & HTML files from {source_path}")
    sources = load_documents(source_path)
    print(sources[2])

    print(f"Splitting {len(sources)} PDF pages in to chunks ...")

    current_file = None
    docs = []
    index = 0
    for index, doc in enumerate(sources):
        filename = doc.metadata["source"].split("/")[-1]
        # print(filename)
        if (
            filename != current_file
            and current_file != None
            or index == len(sources) - 1
        ):
            chunks = split_chunks(
                docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap)
            )
            print(f"Generating index for {current_file} with {len(chunks)} chunks ...")
            generate_index(current_file, chunks, embeddings)
            docs = [doc]
        else:
            docs.append(doc)

        current_file = filename
else:
    print(f"The index persist directory {index_path} is present. Quitting ...")

end = timer()

print(f"Completed in {end - start:.3f}s")
