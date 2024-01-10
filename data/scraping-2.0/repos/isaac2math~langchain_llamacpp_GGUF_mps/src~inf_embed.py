import os
import glob
import yaml

from typing import List
from multiprocessing import Pool
from tqdm import tqdm

from chromadb.config import Settings

from llama_cpp import Llama 

from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

callbacks = BaseCallbackManager([StdOutCallbackHandler()])

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}

# Load DVC parameters
params = yaml.safe_load(open("params.yaml"))["inference"]

embeddings_model_name = params["EMBEDDINGS_MODEL_NAME"]
persist_directory = params['PERSIST_DIRECTORY']
source_directory = params['SOURCE_DIRECTORY']
chunk_size = params['CHUNK_SIZE']
chunk_overlap = params['CHUNK_OVERLAP']

CHROMA_SETTINGS = Settings(
        chroma_db_impl = params['CHROMA_DB_IMPL'],
        persist_directory = persist_directory,
        anonymized_telemetry = params['ANONYMIZED_TELEMETRY'],
)

def load_single_document(file_path: str) -> Document:
    
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:

        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    
    all_files = []
    
    for ext in LOADER_MAPPING:
    
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
    
        results = []
    
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
    
            for i, doc in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
    
                results.append(doc)
                pbar.update()

    return results

def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """

    print(f"Loading documents from {source_directory}")
    print()

    documents = load_documents(source_directory, ignored_files)
    
    if not documents:
        print("No new documents to load")
        exit(0)
    
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    print()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    print()
    
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """

    if os.path.exists(os.path.join(persist_directory, 'index')):
    
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            
            if len(list_index_files) > 3:
            
                return True
    
    return False

def embed():
    # Create embeddings
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
    
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        print()

        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        
        print(f"Creating embeddings. Patience is the greatest warrior...")
        print()

        db.add_documents(texts)
    
    else:
    
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        print()

        texts = process_documents()
        print(f"Creating embeddings. Patience is the greatest warrior...")
        print()

        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    
    db.persist()
    
    db = None

    print(f"Embedding complete")
    print()

if __name__ == "__main__":

    embed()
