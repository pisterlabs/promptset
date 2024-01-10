from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import os
import pinecone

from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv("PINECONE_ENV")

url = "https://lilianweng.github.io/"
loader = RecursiveUrlLoader(url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text)
docs = loader.load()
print(f'Total {len(docs)} pages found.')

text_splitter = TokenTextSplitter(chunk_size=256, chunk_overlap=30)

docs = text_splitter.split_documents(docs)
print(f'Total {len(docs)} chunked documents.')

# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV,  # next to api key in console
)

index_name = "ml-interview-warmup"

# First, check if our index already exists. If it doesn't, we create it
if index_name not in pinecone.list_indexes():
    # we create a new index
    pinecone.create_index(
      name=index_name,
      metric='cosine',
      dimension=384  
)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
Pinecone.from_documents(docs, embedding_function, index_name=index_name)
