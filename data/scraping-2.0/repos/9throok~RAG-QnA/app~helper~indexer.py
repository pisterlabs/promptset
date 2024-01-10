"""
This snippet is used to take the text file and encode it into openai embeddings for retrival.
It is a one time process.

** NO NEED TO RUN AGAIN. THE VECTOR INDEX HAS ALREADY BEEN PREPARED**
"""

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS


raw_documents = TextLoader('./data/sample_data.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
documents = text_splitter.split_documents(raw_documents)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
db.save_local("./indexes/")