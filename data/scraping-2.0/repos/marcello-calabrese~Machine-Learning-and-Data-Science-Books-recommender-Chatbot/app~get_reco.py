
import streamlit as st
###### LLM and Q&A Chatbot functions ######

# Import libraries

from langchain.vectorstores import Chroma
import chromadb
from langchain.embeddings import OpenAIEmbeddings
import openai
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI



@st.cache_resource()    
def create_recommendation(openai_api_key,input_text):
    # load the vector store and embeddings
    persist_directory = 'chroma_db'
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    # Create a retriever
    retriever = vector_db.as_retriever(search_kwargs={'n': 10})
    # create the prompt template
    prompt_template = """ You are an amazon book recommender system. Use the following pieces of context to answer the question and recommend a book to the user.

    if you don't know the answer, just say "I don't know".

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    # create the QA model
    qachain = RetrievalQA.from_chain_type(llm=ChatOpenAI(
                                    openai_api_key=openai_api_key,
                                    model_name = "gpt-3.5-turbo",
                                    temperature=0.1,
                                    verbose=False), 
                                    chain_type='stuff',
                                    chain_type_kwargs={'prompt': PROMPT},
                                    retriever=retriever)
    
    doc_prompt = qachain({"query": input_text})
    
    st.info("🤖 Chatbot: " + doc_prompt['result'])