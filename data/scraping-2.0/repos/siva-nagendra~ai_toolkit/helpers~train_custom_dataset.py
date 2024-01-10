import os
import time
import logging
from . import data_visualizer as dv
from concurrent.futures import ThreadPoolExecutor
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

DATASET_PATH = os.environ["DATASET_PATH"]
DB_FAISS_PATH = os.environ["DB_FAISS_PATH"]

sentence_transformer_model = "all-MiniLM-L6-v2"
start_time = time.time()
doc_count = []
times = [] 

# file types that you want to ingest, including all coding languages
extensions = ['.py', '.java', '.js', '.ts', '.md', '.cpp', '.c', '.cs', '.go', '.rs', '.php', '.html', '.css', '.xml', '.json', '.yaml', '.yml', '.sh', '.rst', '.sql', '.rb', '.pl', '.swift', '.m', '.mm', '.kt', '.gradle', '.groovy', '.scala', '.clj', '.cljs', '.cljc', '.edn', '.lua', '.coffee', 'pdf']

documents = []


def load_file(index: int, file: str) -> List[str]:
    """
    Loads and splits the content of a file.

    Args:
    index (int): The index of the file being processed.
    file (str): The path to the file.

    Returns:
    List[str]: A list of strings representing the split content of the file.
    """
    try:
        loader = TextLoader(file, encoding='utf-8')
        return loader.load_and_split()
    except RuntimeError as e:
        if isinstance(e.__cause__, UnicodeDecodeError):
            try:
                loader = TextLoader(file, encoding='ISO-8859-1')
                return loader.load_and_split()
            except Exception as e_inner:
                logging.exception(f"Failed to load {file} due to error: {e_inner}, Traceback: {e_inner.__traceback__}")
        else:
            logging.exception(f"Failed to load {file} due to an unexpected error: {e}")

    if (index + 1) % 100 == 0:
        logging.info(f"Processed {index + 1} documents.")

    return []


def run_fast_scandir(dir: str, ext: List[str]) -> Tuple[List[str], List[str]]:
    """
    Recursively scans a directory for files with specified extensions.

    Args:
    dir (str): The directory to scan.
    ext (List[str]): A list of file extensions to look for.

    Returns:
    Tuple[List[str], List[str]]: A tuple containing a list of subfolders and a list of files found.
    """
    subfolders, files = [], []
    for f in os.scandir(dir):
        if f.is_dir():
            subfolders.append(f.path)
        if f.is_file():
            if os.path.splitext(f.name)[1].lower() in ext:
                files.append(f.path)

    for dir in list(subfolders):
        sf, f = run_fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def run(DATASET_PATH: str, DB_FAISS_PATH: str) -> None:
    """
    Initiates the process to load and process documents from a dataset, 
    and then save the resultant FAISS database locally.

    Args:
    DATASET_PATH (str): The path to the dataset directory.
    DB_FAISS_PATH (str): The path to save the FAISS database.

    Returns:
    None
    """
    logging.info(f"Training started with\nDATASET_PATH: {DATASET_PATH}\nDB_FAISS_PATH: {DB_FAISS_PATH}")
    subfolders, files = run_fast_scandir(DATASET_PATH, extensions)

    with ThreadPoolExecutor() as executor:
        document_lists = list(executor.map(lambda p: load_file(*p), enumerate(files)))
        documents = [doc for doc_list in document_lists for doc in doc_list if doc_list]

        doc_count.append(len(documents))
        times.append(time.time() - start_time)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    texts = text_splitter.split_documents(documents)

    logging.info(f"Total number of documents pre-processed: {len(documents)}")

    visualizer = dv.DataVisualizer()
    visualizer.start_timer()

    embeddings = HuggingFaceEmbeddings(model_name=f'sentence-transformers/{sentence_transformer_model}',)

    logging.info("Starting the creation of FAISS database from documents...")
    db = FAISS.from_documents(texts, embeddings)
    logging.info("FAISS database created successfully.")

    logging.info(f"Saving the FAISS database locally at {DB_FAISS_PATH}...")
    db.save_local(DB_FAISS_PATH)
    logging.info("FAISS database saved successfully.")

    visualizer.generate_plots(documents=files)
