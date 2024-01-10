import asyncio
import os
import pandas as pd
# import textract
import matplotlib.pyplot as plt
from dotenv import load_dotenv 
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain  

load_dotenv()
  
print("Loading dataset...")
 
# Simple method - Split by pages 
loader = PyPDFLoader("./tokyo_api_reference_7-12-2023.pdf")
pages = loader.load_and_split()
chunks = pages

# Get embedding model
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)


# Create conversation chain that uses our vectordb as retriver, this also allows for chat history management
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.7, max_tokens=1024), db.as_retriever())

print("Ready")

class Translator:
    async def translate(self, text: str) -> str:
        # Implement the translation logic here
        # Return the translated text
        result = qa({"question": text, "chat_history": []})
 
        return result
