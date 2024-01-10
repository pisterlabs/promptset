import chromadb
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

import os
from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env

loader = TextLoader("some_data/FDR_State_of_Union_1944.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding_function,persist_directory='./speech_embedding_db')
db.persist()
db_connection = Chroma(persist_directory='./speech_embedding_db/',embedding_function=embedding_function)

# WATCH THE VIDEO TO TRULY UNDERSTAND WHY YOU MAY NOT WANT TO DO DIRECT QUESTIONS!
new_doc = "What did FDR say about the cost of food law?"
docs = db_connection.similarity_search(new_doc)
print(docs[0].page_content)

# load the document and split it into chunks
loader = TextLoader("some_data/Lincoln_State_of_Union_1862.txt")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function,persist_directory='./speech_embedding_db')
docs = db.similarity_search('slavery')
print(docs[0].page_content)