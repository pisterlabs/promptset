import os
import pickle
import shutil
from llama_index import download_loader
import hashlib

from multiprocessing import Lock
from multiprocessing.managers import BaseManager
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, ServiceContext, StorageContext, load_index_from_storage
from llama_index import LangchainEmbedding, LLMPredictor
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

index = None
stored_docs = {}
doc_hashes = {'titles': set(), 'contents': set()}
lock = Lock()
DOCUMENTS_DIR = "./documents"

def compute_hash(data):
    """Compute the SHA-256 hash of the given data."""
    return hashlib.sha256(data.encode()).hexdigest()

def ensure_unique_filename(directory, filename):
    """Ensure that the filename is unique within the given directory.
    If a file with the same name already exists, append a counter to it."""
    name, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{name}_{counter}{ext}"
        counter += 1

    return unique_filename

index_name = "./saved_index"
pkl_name = "stored_documents.pkl"

def initialize_index():
    global index, stored_docs, doc_hashes

    if not os.path.exists(DOCUMENTS_DIR):
        os.mkdir(DOCUMENTS_DIR)
    
    api_type = os.environ.get('OPENAI_API_TYPE')
    if api_type == "azure":
        llm = AzureChatOpenAI(
            deployment_name=os.environ.get("OPENAI_API_LLM", "gpt-35-turbo"), 
            openai_api_version=os.environ.get("OPENAI_API_VERSION", "2023-05-15"), 
            temperature=os.environ.get("OPENAI_API_TEMPERATURE", 0.5))
        embeddings = LangchainEmbedding(OpenAIEmbeddings(deployment=os.environ.get("OPENAI_API_EMBEDDING", "text-embedding-ada-002")))
        llm_predictor = LLMPredictor(llm=llm, system_prompt=os.environ.get("OPENAI_API_PROMPT", "You are a helpful assistant of CIPPlanner."))
        service_context = ServiceContext.from_defaults(chunk_size_limit=512, embed_model=embeddings, llm_predictor=llm_predictor)
    else:
        service_context = ServiceContext.from_defaults(chunk_size_limit=512)

    with lock:
        if os.path.exists(index_name):
            index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name), service_context=service_context)
        else:
            index = GPTVectorStoreIndex([], service_context=service_context)
            index.storage_context.persist(persist_dir=index_name)

        if os.path.exists(pkl_name):
            with open(pkl_name, "rb") as f:
                stored_docs, doc_hashes = pickle.load(f)
        else:
            stored_docs = {}
            doc_hashes = {'titles': set(), 'contents': set()}

def query_index(query_text):
    global index
    response = index.as_query_engine().query(query_text)
    return response

def insert_into_index(doc_file_path, doc_id=None):
    global index, stored_docs, doc_hashes

    if doc_file_path.endswith(".xlsx"):
        reader = download_loader("PandasExcelReader")
        loader = reader(pandas_config={"header": 0})
        document = loader.load_data(doc_file_path)[0]
    else:
        document = SimpleDirectoryReader(input_files=[doc_file_path]).load_data()[0]

    if doc_id is not None:
        document.doc_id = doc_id

    doc_title_hash = compute_hash(document.doc_id)
    doc_content_hash = compute_hash(document.text)

    if doc_title_hash in doc_hashes.get('titles', set()):
        raise ValueError("Document with similar title already exists!")
    if doc_content_hash in doc_hashes.get('contents', set()):
        raise ValueError("Document with similar content already exists!")

    doc_hashes.setdefault('titles', set()).add(doc_title_hash)
    doc_hashes.setdefault('contents', set()).add(doc_content_hash)
    
    # Save the actual document file to the documents folder
    original_filename = os.path.basename(doc_file_path)
    unique_filename = ensure_unique_filename(DOCUMENTS_DIR, original_filename)
    shutil.copy(doc_file_path, os.path.join(DOCUMENTS_DIR, unique_filename))

    with lock:
        stored_docs[document.doc_id] = document.text
        index.insert(document)
        index.storage_context.persist(persist_dir=index_name)
        
        try:
            with open(pkl_name, "wb") as f:
                pickle.dump((stored_docs, doc_hashes), f)
            print(f"Successfully saved {document.doc_id} to {pkl_name}.")
        except Exception as e:
            print(f"Error while saving {document.doc_id} to {pkl_name}: {str(e)}")

    return

def get_documents_list():
    global stored_docs
    documents_list = []
    for doc_id, doc_text in stored_docs.items():
        documents_list.append({"id": doc_id, "text": doc_text})
    return documents_list

if __name__ == "__main__":
    print("initializing index...")
    initialize_index()

    manager = BaseManager(('', 4003), b'password')
    manager.register('query_index', query_index)
    manager.register('insert_into_index', insert_into_index)
    manager.register('get_documents_list', get_documents_list)
    server = manager.get_server()

    print("server started...")
    server.serve_forever()
