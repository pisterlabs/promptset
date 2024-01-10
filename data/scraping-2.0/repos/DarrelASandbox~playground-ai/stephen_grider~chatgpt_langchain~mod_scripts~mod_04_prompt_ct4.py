"""
A script for retrieval-based question answering using the langchain library.

This script demonstrates how to integrate a retrieval system with a chat model for answering questions.
It utilizes Chroma for retrieval of relevant information and ChatOpenAI for 
generating answers based on the retrieved content. 
This setup is ideal for answering questions with context from a specific knowledge base.

Features:
- Initialize ChatOpenAI for language model-based interactions.
- Use OpenAI embeddings for document retrieval.
- Load a Chroma database for document retrieval based on embeddings.
- Set up a RetrievalQA chain combining the chat model and the retriever.
- Answer a specific question using the RetrievalQA chain.

Usage:
Run the script to ask a question about the English language and get an answer based on 
retrieved content from the Chroma database.
"""

import langchain
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

langchain.debug = True

load_dotenv()

# Initialize a ChatOpenAI instance for language model interactions.
chat = ChatOpenAI()

# Initialize OpenAI embeddings for document retrieval.
embeddings = OpenAIEmbeddings()

# Load a Chroma database for document retrieval.
db = Chroma(persist_directory="emb", embedding_function=embeddings)

# Set up the retriever using the Chroma database.
retriever = db.as_retriever()

# Configure the RetrievalQA chain with the chat model and the retriever.
# https://python.langchain.com/docs/modules/chains/document/
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="refine")

# Run the chain to answer a question based on retrieved content.
result = chain.run("What is an interesting fact about the English language?")
print(result)
