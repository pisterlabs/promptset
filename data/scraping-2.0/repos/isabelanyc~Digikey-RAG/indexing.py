import os
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# load all the pdfs in the data directory
# type of loader is the UnstructuredPDFLoader
print('Loading files...')
loader = DirectoryLoader('data/txt', glob="**/*.txt", show_progress=True)

# create a text splitter and chunk up the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap  = 1000,
    length_function = len,
    add_start_index = True,
)
docs = loader.load_and_split(text_splitter)

num_docs = len(docs)
print(f'Number of docs: {num_docs}')

# Use Chroma to create the embdeddings, create the vector store and save to disk
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
db = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory="chroma_db")
