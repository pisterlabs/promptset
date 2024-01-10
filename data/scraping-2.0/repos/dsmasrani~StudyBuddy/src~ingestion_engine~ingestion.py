import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import OPENAPI_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, OnlinePDFLoader
import tiktoken
from uuid import uuid4
from tqdm.auto import tqdm
import hashlib
from docx2pdf import convert

#Batch limit of upload size (can go upto 1000)
batch_limit = 100

#Helper function to calculae length of TOKENS not characters
def tiktoken_len(text):
    tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokenizer = tiktoken.get_encoding('cl100k_base')

    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

#Splits each chunk of text by token length to help ChatGPT
def text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter

#Compute Hash
def compute_md5(text):
    m = hashlib.md5()
    m.update(text.encode('utf-8'))
    return m.hexdigest()

#Intializes the pinecone index and uploads the embeddings
##TODO: Split this function to improve readability
def initalize_embeddings(data, VERBOSE):
    texts = []
    metadatas = []
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAPI_KEY,
    )

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
    )

    index = pinecone.GRPCIndex(PINECONE_INDEX_NAME)
    
    print(index.describe_index_stats()) if VERBOSE else None
    txt_splitter = text_splitter()
    texts = []
    metadatas = []
    for i, document in enumerate(tqdm(data)): #For each page in the document
        metadata = {
            'source': document.metadata['source'],
            'page': document.metadata['page'] + 1,}
        record_texts = txt_splitter.split_text(document.page_content)
        record_metadatas = [{
            "chunk": j, "text": chunk, 'source': (document.metadata['source'].split('/')[-1] + ' Page: ' + str(document.metadata['page'])) 
        } for j, chunk in enumerate(record_texts)] #Each page will be associated with a metadata

        texts.extend(record_texts)
        metadatas.extend(record_metadatas)
    for i in tqdm(range(0,len(texts),batch_limit)):
        text_tmp = texts[i:i+batch_limit]
        metadata_tmp = metadatas[i:i+batch_limit]
        ids = [compute_md5(text_tmp[i]) for i in range(len(text_tmp))]
        embeds = embed.embed_documents(text_tmp)
        index.upsert(vectors=zip(ids, embeds, metadata_tmp))