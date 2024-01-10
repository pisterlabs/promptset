import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

loader = SeleniumURLLoader(urls=[sys.argv[1]])
data = loader.load()                                                                                                                        
print(f"[1] Successfully loaded {len(data)} url(s).")

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
texts = text_splitter.split_documents(data)
print(f"[2] Successfully Split into {len(texts)} doc(s).")

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_API_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
index_name = "61b-chain"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

vector_store = Pinecone.from_existing_index(index_name, embeddings, "website_data")
docsearch = vector_store.add_documents(texts, namespace=sys.argv[2])
print(f"[3] Successfully uploaded documents to Pinecone.")