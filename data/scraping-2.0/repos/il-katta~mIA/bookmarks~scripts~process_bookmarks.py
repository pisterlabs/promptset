import json
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import config

with open(config.DATA_DIR / 'bookmarks.json', 'r') as f:
    docs = [Document(**d) for d in json.load(f)]

print(f"Documents: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(docs)

print(f"Chunks: {len(all_splits)}")
persist_directory = str(config.DATA_DIR / "bookmarks_vectorstore")
embedding = OpenAIEmbeddings(show_progress_bar=True)
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
collection_name = "bookmarks"
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
    collection_name=collection_name,
    persist_directory=persist_directory,

)
vectorstore.persist()

vectorstore = None
del vectorstore

vectorstore: Optional[Chroma] = Chroma(
    persist_directory=persist_directory,
    collection_name=collection_name,
    embedding_function=embedding
)
docs = vectorstore.similarity_search("python code")

for doc in docs:
    print(doc.metadata["source"])
