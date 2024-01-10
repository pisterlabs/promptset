import os
import streamlit as st
import random
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate

OPENAI_API_KEY = st.secrets["OPENAI_API_SECRET"]

def InitializeVectorStore():
  # Load each document from the data folder
  raw_documents = [TextLoader(os.path.join("data", filename)).load() for filename in os.listdir("data") if filename.endswith(".txt")]
  raw_documents += [PyPDFLoader(os.path.join("data", filename)).load_and_split() for filename in os.listdir("data") if filename.endswith(".pdf")] 
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  documents = [text_splitter.split_documents(document) for document in raw_documents]

  # expand the documents list of lists into one list
  documents = [document for sublist in documents for document in sublist]

  # Create a vector store for the documents
  db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
  retriever = db.as_retriever()
  return retriever


def InvokeInterviewQuestion(question: str, retriever):
  # Initialize messages for the chat model
  system_template = """
    Your name is Alex.  You are answering questions in an informal interview.  
    You are to answer questions in the first-person and keep a positive tone.  You are to answer questions professionally and eloquently.  
    You are not to acknlowedge that you are an AI in any regard.  You may state that you are an assistant if necessary.
    Use available context to provide a compelling answer.  
    If a question is unrelated to an interview question, please do not answer it.  If you do not know the answer to a question, please say so.
    You will see a question and a context.  You should use the context to answer the question.  An example will look like:
    ```
    You should use the following context for answering the subsequent question:
    [context]

    Question: [question]
    ```
    """
  system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

  human_template = """
    You should use the following context for answering the subsequent question:
    {context}
    
    Question: {question}
    """
  human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

  # Create the chat model chain
  chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    | ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    | StrOutputParser()
  )

  for chunk in chain.stream(question):
    yield chunk
