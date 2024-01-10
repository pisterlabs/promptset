import os
import pinecone

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment="gcp-starter")

embeddings = OpenAIEmbeddings()


data = Pinecone.from_texts(texts=["./assets/economia.txt",
                                  "./assets/ingenieria-civil.txt",
                                  "./assets/ingenieria-electrica.txt",
                                  "./assets/ingenieria-electronica.txt",
                                  "./assets/ingenieria-sistemas.txt",
                                  "./assets/ingenieria-industrial.txt"],
embedding=embeddings,
index_name="challenge1")
