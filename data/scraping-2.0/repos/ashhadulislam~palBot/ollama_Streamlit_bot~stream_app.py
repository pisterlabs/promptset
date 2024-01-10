# from langchain import PromptTemplate, LLMChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain.llms import CTransformers
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from io import BytesIO
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import GPT4AllEmbeddings
import streamlit as st
from pathlib import Path
import os
from langchain import hub
import streamlit as st
from langchain.llms import Ollama
import pandas as pd

from langchain.chat_models import ChatOpenAI
# os.environ["OPENAI_API_KEY"] = ''
# import lib
from PyPDF2 import PdfWriter

from PyPDF2 import PdfReader
import base64
QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")


def load_llm():
	llm = Ollama(
	model="zephyr",
	verbose=True,	
	)
	return llm

@st.cache_resource
def prepare_llm(chosen_resource,chosen_file):

    llm = load_llm()

    
    embeddings=GPT4AllEmbeddings()
    
    
    load_vector_store = Chroma(persist_directory=f"vectorstores/{chosen_resource}/{chosen_file}", 
                               embedding_function=embeddings)
    retriever = load_vector_store.as_retriever(search_kwargs={"k":2})

    return llm,retriever


    
def get_response(input):
    prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

    
    chain_type_kwargs = {"prompt": prompt}


    
    query = input
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                   retriever=retriever, return_source_documents=True, 
                                   chain_type_kwargs=chain_type_kwargs, verbose=True)
    response = qa(query)
    return response



resource_folders=os.listdir("vectorstores")
if ".DS_Store" in resource_folders:
    resource_folders.remove(".DS_Store")

st.title("PalestineBot - Your Personal Chatbot for all things Palestine")
    
for folder in resource_folders:
    print(folder)
chosen_resource=st.selectbox("Choose consultation resource", resource_folders, index=0)

document_files=os.listdir(f"vectorstores/{chosen_resource}")
if ".DS_Store" in document_files:
    document_files.remove(".DS_Store")
    


    
for file in document_files:
    print(file)
    
chosen_file=st.selectbox("Choose consultation file", document_files, index=0)    
print(chosen_resource,chosen_file)
llm,retriever=prepare_llm(chosen_resource,chosen_file)    


Link_list=["ElIntifada"]

df_link_file = pd.DataFrame()

if chosen_resource in Link_list:
    df_link_file=pd.read_csv(f"resources/{chosen_resource}/{chosen_file}/link_match.csv")
    print(df_link_file.head())

st.subheader("I have studied the document")
st.subheader(chosen_file)
st.subheader("From: "+chosen_resource)
st.header("Ask me anything about it")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        
response=None
# React to user input
if prompt := st.chat_input("What is your query?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})    
    
    
    resp=get_response(prompt)    
    print("response is ",resp)
    reference=resp["source_documents"][0].page_content
    reference=reference.replace("\n"," ")
    page=""
    if page in resp["source_documents"][0].metadata:
        page=resp["source_documents"][0].metadata["page"]
    file=""
    if "source" in resp["source_documents"][0].metadata:
        file=resp["source_documents"][0].metadata["source"]
        the_link=None
        file_stripped=file.split("/")
        if len(file_stripped)>=4:
            just_file=file_stripped[3]
            if not df_link_file.empty:
                row=df_link_file[df_link_file["file_name"]==just_file]
                if row.shape[0]>0:
                    the_link=list(row["link"])[0]
    
    response = f"{resp['result']}\n\n *Page: {page}\n\n *File:{file}"
    if the_link:
        response=response+f"\n\nRead details [here]({the_link})"
    

#     response=prompt
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        # if the_link:
        #     st.write(f"Read details [here]({the_link})")
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

        
