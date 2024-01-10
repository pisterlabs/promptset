from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.schema import Document
from langchain import OpenAI
import langchain
import os
import glob
from dotenv import load_dotenv
import openai
import constants
import time

# Set OpenAI API Key
load_dotenv()
os.environ["OPENAI_API_KEY"] = constants.APIKEY
openai.api_key = constants.APIKEY 

# Set paths
source_path = './data/src/'
store_path = './vectorstore/'
destination_file = './data/ingested.txt'

# Load documents
print("===Loading documents===")
text_loader_kwargs={'autodetect_encoding': True}
loader = DirectoryLoader(source_path,
                         show_progress=True,
                         use_multithreading=True,
                         loader_cls=TextLoader,
                         loader_kwargs=text_loader_kwargs)
documents = loader.load()

# Split documents
print("===Splitting documents into chunks===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap  = 150,
    length_function = len,
    add_start_index = True,
)

split_documents = text_splitter.split_documents(documents)

# Embedding documents
print("===Embedding text and creating database===")
embeddings = OpenAIEmbeddings(
    show_progress_bar=True,
    request_timeout=60,
)

db = FAISS.from_documents(split_documents, embeddings)
db.save_local(store_path, "index")

# Record what we have ingested
print("===Recording ingested files===")
with open(destination_file, 'w') as f:
    for document in documents:
        f.write(os.path.basename(document.metadata['source']))
        f.write('\n')
            
