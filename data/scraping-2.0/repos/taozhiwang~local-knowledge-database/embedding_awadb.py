from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import openai
import awadb
from langchain.vectorstores import AwaDB
import os

#print(awadb.__version__)

loader = TextLoader('./data/state_of_the_union.txt', encoding='utf8')
data = loader.load()

# Create split file
text_splitter = CharacterTextSplitter(chunk_size=40, chunk_overlap=5)
split_docs = text_splitter.split_documents(data)

embeddings = HuggingFaceEmbeddings()

db = AwaDB.from_documents(,)
#db.save_local("awadb_index")

awadb_client = awadb.Client()
awadb_client.Create("test_llm1") 

print(split_docs)

for text in split_docs:
    awadb_client.Add([split_docs[0].page_content])