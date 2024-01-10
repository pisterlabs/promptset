import os
import asyncio
import chainlit as cl
from langchain.prompts import (
    PromptTemplate,
)
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import chromadb
from chromadb.config import Settings
DB_DIR = "./db"

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
LLM_MODEL_NAME = "text-davinci-003"              # OpenAI
EMBEDDING_MODEL_NAME = "text-embedding-ada-002"  # OpenAI
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# create a prompt template
prompt_template = """
You are a helpful assistant that truthfully respond to a user's query about 
the books Art of War or the Prince.

User's query: {query}

If you don't know the answer, simply answer: I don't know. 
Most importantly, do not respond with false information.
"""

prompt = PromptTemplate(
    input_variables=['query'],
    template=prompt_template
)

@cl.on_message
def main(query: str):
    retriever = None
    embeddings = OpenAIEmbeddings(openai_api_key = os.environ['OPENAI_API_KEY'],
                                  model=EMBEDDING_MODEL_NAME)

    if not os.path.exists(DB_DIR):
        # digest the texts into chunks & store their embeddings
        loader = DirectoryLoader(path="../data/", glob="**/*.txt")
        docs = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
        text_chunks = text_splitter.split_documents(documents=docs)
        vectordb = Chroma.from_documents(text_chunks, embeddings, 
                                          persist_directory="./db")
    else:
        # lookup from existing stored embeddings
        vectordb = Chroma(persist_directory=DB_DIR, 
                          embedding_function=embeddings)

    retriever = vectordb.as_retriever(search_type="mmr") # maximal margin R
    qa = RetrievalQA.from_chain_type(llm = OpenAI(model=LLM_MODEL_NAME,
                                                  temperature=0.0),
                                     chain_type="stuff",
                                     retriever=retriever,
                                     return_source_documents=True
                                    )

    try:
        answer = qa({
            "query": query
        })
        
        response = f"{answer['result']}\n"
        for doc in answer['source_documents']:
            tabbed_content = doc.page_content.replace("\n", "")
            response += f"\n\t{doc.metadata['source']}: {tabbed_content[:60]}"
    except Exception as e:
        response = f"I don't know.  Please ask another question. {e}"
    asyncio.run(
        cl.Message(
            content=response
        ).send()
    )

   
@cl.on_chat_start
def start():
    asyncio.run(
        cl.Message(
            content="Ask me anything about The Prince or the Art of War!"
        ).send()
    )
