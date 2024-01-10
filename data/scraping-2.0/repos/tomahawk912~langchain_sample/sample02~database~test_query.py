import os

from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

load_dotenv()

# CUR_DIR = os.path.dirname(os.path.abspath(__file__))

# CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
CHROMA_PERSIST_DIR = r"D:\Langchain\part06\ch03_langchain\gen3\database\chroma-persist"
CHROMA_COLLECTION_NAME = "tomahawk912-bot"

from pprint import pprint

db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)

# 사용 추천 (옵션 많음)
docs = db.similarity_search("i want to know about planner of semantic kernel", k=10)

for i, doc in enumerate(docs):
    print(f"Document {i} ================ \n")
    pprint(doc.page_content)
    print(doc.metadata)
