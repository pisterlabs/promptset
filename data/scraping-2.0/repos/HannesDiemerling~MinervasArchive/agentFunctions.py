import prompts
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import os
import streamlit as st

def tearm_search(user_input):

    return st.session_state.reportvectorstore.similarity_search(query=user_input,k=3)

def simple_report_search(user_input):

    summarizer = AzureChatOpenAI(request_timeout=30, temperature=0.1, model="summarizer", deployment_name=os.getenv("OPENAI_MODERATOR_NAME"))
    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(summarizer,retriever=st.session_state.reportvectorstore.as_retriever(search_kwargs={"k": 2}))
    
    result = qa_chain({"question": user_input,"handle_parsing_errors":True}, return_only_outputs=True)

    return result

def one_person_search(user_input):
    
    summarizer = AzureChatOpenAI(request_timeout=30, temperature=0.1, model="summarizer", deployment_name=os.getenv("OPENAI_MODERATOR_NAME"))
    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(summarizer,retriever=st.session_state.personvectorstore.as_retriever(search_kwargs={"k": 2}))
    
    result = qa_chain({"question": user_input,"handle_parsing_errors":True}, return_only_outputs=True)

    return result

def report_summarizer(user_input):

    summarizer = AzureChatOpenAI(request_timeout=30, temperature=0.1, model="summarizer", deployment_name=os.getenv("OPENAI_MODERATOR_NAME"))
    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(summarizer,retriever=st.session_state.reportvectorstore.as_retriever(search_kwargs={"k": 5}))
    
    result = qa_chain({"question": user_input,"handle_parsing_errors":True}, return_only_outputs=True)
    
    return result
