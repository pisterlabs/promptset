from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from src.config import settings

connection_string = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=settings.postgres_host,
    port=settings.postgres_port,
    database=settings.postgres_vector_db,
    user=settings.postgres_user,
    password=settings.postgres_password)

ollama_url=f'http://{settings.ollama_host}:{settings.ollama_port}'
model = settings.ollama_model
embeddings = OllamaEmbeddings(base_url = ollama_url, model = model )

async def vectorize(collection_name: str, text: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    await PGVector.afrom_texts(embedding=embeddings,
                         texts=texts,
                         collection_name=collection_name,
                         connection_string=connection_string)
