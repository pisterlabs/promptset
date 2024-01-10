

import environment

import os
import getpass
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from llms import defaultLLM as llm
from embeddings import defaultEmbeddings as embedding

# os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')
# os.environ['ACTIVELOOP_TOKEN'] = getpass.getpass('Activeloop Token:')
# os.environ['ACTIVELOOP_ORG'] = getpass.getpass('Activeloop Org:')

org = os.environ['ACTIVELOOP_ORG']
# embeddings = OpenAIEmbeddings()
dataset_path = 'hub://' + org + '/data'

with open("./documents/messages.txt") as f:
    state_of_the_union = f.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
pages = text_splitter.split_text(state_of_the_union)
pages = [Document(page_content=doc) for doc in pages]
print(pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(pages)
# texts = [Document(page_content=doc) for doc in texts]

print (texts)


db = DeepLake.from_documents(texts, embedding, dataset_path=dataset_path, overwrite=True)


db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embedding)

retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False)

# What was the restaurant the group was talking about called?
query = input("Enter query:")

# The Hungry Lobster
ans = qa({"query": query})

print(ans)