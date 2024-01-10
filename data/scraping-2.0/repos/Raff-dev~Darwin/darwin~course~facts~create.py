import os
import sys

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=200, chunk_overlap=0)

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY)  # type: ignore

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="embeddings")
