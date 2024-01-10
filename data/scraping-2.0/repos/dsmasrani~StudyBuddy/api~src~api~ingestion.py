from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from api.src.api import auth
import sqlalchemy
from api.src import database as db
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tiktoken
from tqdm.auto import tqdm
import hashlib
import logging
import httpx
import os
import dotenv
from supabase import create_client, Client

url: str = os.environ.get("PROJECT_URL")
key: str = os.environ.get("PROJECT_KEY")
supabase: Client = create_client(url, key)

batch_limit = 100
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
router = APIRouter(
    prefix="/ingestion",
    tags=["ingestion"],
    dependencies=[Depends(auth.get_api_key)],
)    


@router.get("/retrieve_files")
def retrieve_files():
    """"""    
    with db.engine.begin() as connection:
        result = connection.execute(sqlalchemy.text("SELECT name from objects ORDER BY created_at ASC")).fetchall()

    files = [{"name": row[0]} for row in result]

    return files

@router.post("/resolve_queue")
def resolve_queue(background_tasks: BackgroundTasks):
    files = retrieve_files()
    logging.debug("Retrieved list of files")
    
    background_tasks.add_task(backgroundProcessing, files)
    return {"message": "Queue processing started successfully"}

def backgroundProcessing(files):
    for file in files:
        file_name = file['name']
        if ".emptyFolderPlaceholder" in file_name:
            logging.debug("Skipping file %s", file_name)
            continue
        logging.debug("Processing file %s", file_name)
        signed_file_url = supabase.storage.from_('files').create_signed_url(file_name, 3600)
        logging.debug("Public URL: %s", signed_file_url)
        file_url = signed_file_url['signedURL']
        user_UUID = file_name.split('/')[0]
        logging.debug("User UUID: %s", user_UUID)
        try:
            with db.engine.begin() as connection:
                user_keys_query = sqlalchemy.text("SELECT * FROM user_keys WHERE user_id = :user_UUID")
                user_keys_result = connection.execute(user_keys_query, {'user_UUID': user_UUID}).fetchone()
        except:
            logging.error("Invalid Credentials.")
            raise HTTPException(403, "Invalid Credentials")
        user_email = user_keys_result.user_email
        logging.debug("Starting Ingestion")
        process_file(file_url, user_email, file_name)
        supabase.storage.from_('files').remove(file_name)
    logging.debug("queue resolved successfully (or empty)")
    print("Queue resolved successfully (or empty)")
    return 200

@router.post("/process_file")
def process_file(file_url: str, user_email: str, file_name: str = None):
    """"""
    logging.debug("Getting Credentials")
    with db.engine.begin() as connection:
        user_keys_query = sqlalchemy.text("SELECT * FROM user_keys WHERE user_email = :user_email")

        user_keys_result = connection.execute(user_keys_query, {'user_email': user_email}).fetchone()
        #print(user_keys_result)

        if not user_keys_result:
            logging.error("Invalid Credentials. Please check your email and try again.")
            raise HTTPException(400, "Invalid Credentials. Please check your email and try again.")
        
        pinecone_key = user_keys_result.pinecone_key
        pinecone_env = user_keys_result.pinecone_env
        index_name = user_keys_result.index_name
        openai_key = user_keys_result.openai_key
        logging.debug("Credentials retrieved successfully")
        logging.debug("Pinecone Key: %s, Pinecone Env: %s, Index Name: %s, OpenAI Key: %s", pinecone_key, pinecone_env, index_name, openai_key)

        logging.debug("Starting Ingestion")
        run_ingestion(pinecone_key, pinecone_env, index_name, openai_key, user_email, file_url, file_name)

    return {"message": "Ingestion initialized successfully"}

def run_ingestion(pinecone_key, pinecone_env, index_name, openai_key, user_email, file_url, file_name):
    """ """
    #uri = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    #uri = somefunction()
    logging.debug("Downloading PDF from %s", file_url)
    try:
        loader = PyPDFLoader(file_url)
    except:
        logging.error("Invalid URL. Please check your URL and try again.")
        raise HTTPException(400, "Invalid URL. Please check your URL and try again.")
    
    logging.debug("Loading PDF to memory (This might take a while)...")
    data = loader.load() #Data is an an array of Document objects with each object having a page_content and metadata
    logging.debug("PDF loaded successfully")
    return initalize_embeddings(pinecone_key, pinecone_env, index_name, openai_key, data, user_email, file_name)


def initalize_embeddings(pinecone_key, pinecone_env, index_name, openai_key, data, user_email, file_name):
    texts = []
    metadatas = []
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=openai_key,
    )

    pinecone.init(
        api_key=pinecone_key,
        environment=pinecone_env,
    )

    index = pinecone.GRPCIndex(index_name)
    txt_splitter = text_splitter()
    texts = []
    metadatas = []
    logging.debug("Splitting PDF into chunks and injecting metadata...")
    for i, document in enumerate(tqdm(data)): #For each page in the document
        metadata = {
            'source': document.metadata['source'] if not file_name else file_name,
            'page': document.metadata['page'] + 1,}
        record_texts = txt_splitter.split_text(document.page_content)
        record_metadatas = [{
            "chunk": j, "text": chunk, 'source': ((document.metadata['source'].split('/')[-1] if not file_name else file_name) + ' Page: ' + str(document.metadata['page'])) 
        } for j, chunk in enumerate(record_texts)] #Each page will be associated with a metadata
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
    #uploading to pinecone
    logging.debug("Uploading to Pinecone...")
    for i in tqdm(range(0,len(texts),batch_limit)):
        text_tmp = texts[i:i+batch_limit]
        metadata_tmp = metadatas[i:i+batch_limit]
        ids = [compute_md5(text_tmp[i]) for i in range(len(text_tmp))]
        embeds = embed.embed_documents(text_tmp)
        index.upsert(vectors=zip(ids, embeds, metadata_tmp))
    logging.debug("Ingestion completed successfully")
    return {"message": f"Ingestion uploaded successfully for {user_email}"}

#Splits each chunk of text by token length to help ChatGPT
def text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter

#Helper function to calculae length of TOKENS not characters
def tiktoken_len(text):
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def compute_md5(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()