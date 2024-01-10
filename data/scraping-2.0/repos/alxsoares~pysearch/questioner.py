import pandas as pd
import tiktoken
import faiss
import numpy as np
from ai import AI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from faiss import write_index, read_index

ai = AI()
embeddings = ai.get_open_ai_embedding()
db = FAISS.load_local("houses_new.index", embeddings)

#response = db.similarity_search_with_score("List homes that have more than 4 bedrooms?", filter=dict(page=1))

#print(response)
#for doc, score in response:
#    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

results = db.similarity_search("List me homes with best value for money?", k=1, fetch_k=4)
for doc in results:
    print(f"Content: {doc.page_content}, Metadata: {doc.metadata}")
