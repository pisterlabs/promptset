import os
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

_ = load_dotenv(find_dotenv())  # read local .env file

root_dir = "./source"

docs = []

for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

print(f"{len(docs)}")


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cuda"}
)

# Create and store locally vectorstore
print("Creating new vectorstore")
print(f"Creating embeddings. May take some minutes...")
persist_directory = "db"
chroma_settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=persist_directory,
    anonymized_telemetry=False,
)
db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory=persist_directory,
    client_settings=chroma_settings,
)
db.persist()
