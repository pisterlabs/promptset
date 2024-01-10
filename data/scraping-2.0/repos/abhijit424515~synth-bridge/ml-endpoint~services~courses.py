import chromadb
import pandas as pd
from langchain.vectorstores.chroma import Chroma
import os
from dotenv import load_dotenv
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv("../.env")

df = pd.read_csv('data/course.csv')
url = {title: url for title, url in zip(df["title"], df["url"])}

# Instance of ChromaDB with OpenAI's embedding function
langchain_chroma = Chroma(
    collection_name="courses",
    client=chromadb.PersistentClient(path="./course-db"),
    embedding_function=OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
)

# Find the most relevant courses using similarity search on the embeddings
def predict_courses(query):
  docs = langchain_chroma.similarity_search(query, k = 3)    
  return list(map(lambda x: {"title": df["title"][int(x.metadata["index"])], "description": x.page_content[:-1], "link": url[df["title"][int(x.metadata["index"])]]}, docs))
