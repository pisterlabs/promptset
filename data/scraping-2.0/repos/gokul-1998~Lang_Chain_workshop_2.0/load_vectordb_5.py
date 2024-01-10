from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from env import OPENAI_API_KEY, QDRANT_URL, qdrant_Api_Key

loader = TextLoader("./data/insurance.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
all_splits = text_splitter.split_documents(documents)

print(len(all_splits))
Qdrant.from_documents(
  documents=all_splits,
  embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"),
  url=QDRANT_URL,
  collection_name='insurance',
  api_key=qdrant_Api_Key
)