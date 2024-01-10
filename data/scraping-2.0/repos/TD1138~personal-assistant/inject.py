import glob
import os

from pa.constants import CHROMA_SETTINGS, PERSIST_DIRECTORY, SOURCE_DIRECTORY

from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PDFMinerLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader
    )
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings


LOADER_MAPPING = {
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}


def load_single_document(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str):
    # Loads all documents from source documents directory
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_single_document(file_path) for file_path in all_files]


def main():
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(
                                               chunk_size=1000, 
                                               chunk_overlap=200)

    texts = text_splitter.split_documents(documents)
    
    instruction = "Represent the document for retrieval; Input: " # same as in the paper
    
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                      model_kwargs={"device": "cuda"},
                                                      embed_instruction=instruction)
    
    # Create and store locally vectorstore
    db = Chroma.from_documents(texts, instructor_embeddings, 
                           persist_directory=PERSIST_DIRECTORY, 
                           client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

if __name__ == "__main__":
    main()
