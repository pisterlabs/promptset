import torch
import subprocess
import streamlit as st
from   run_redhatai import load_model
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME, SOURCE_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

import os
import shutil
from streamlit_extras.stateful_chat import chat, add_message
from streamlit_extras.app_logo import add_logo

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIRECTORY = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS"
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"




def model_memory():
    # Adding history to the model.
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer,\
    just say that you don't know, don't try to make up an answer.

    {context}

    {history}
    Question: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return prompt, memory


    


DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_session_result_state():
    if "result" not in st.session_state:
    # Run the document ingestion process. 
        if os.path.exists(PERSIST_DIRECTORY):
            try:
                print("The directory  exist , making a fresh one")
                shutil.rmtree(PERSIST_DIRECTORY)
            except OSError as e:
                print(f"Error: {e.filename} - {e.strerror}.")
        else:
            print("The directory does not exist")
        run_langest_commands = ["python", "ingest.py"]
        run_langest_commands.append("--device_type")
        run_langest_commands.append(DEVICE_TYPE)

        result = subprocess.run(run_langest_commands, capture_output=True)
        st.session_state.result = result

# Define the retreiver
# load the vectorstore

    if "EMBEDDINGS" not in st.session_state:
        EMBEDDINGS = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": DEVICE_TYPE})
        st.session_state.EMBEDDINGS = EMBEDDINGS

    if "DB" not in st.session_state:
       
        DB = Chroma(persist_directory=PERSIST_DIRECTORY,embedding_function=st.session_state.EMBEDDINGS,client_settings=CHROMA_SETTINGS)
        if DB is None:
            print("Failed to initialize DB in initialize_session_db_state")
        st.session_state.DB = DB
        

   
 
def initialize_session_qa_state():
    if "RETRIEVER" not in st.session_state:
        db = st.session_state.get('DB')
        RETRIEVER = db.as_retriever()
        st.session_state.RETRIEVER = RETRIEVER

    if "LLM" not in st.session_state:
        LLM = load_model(device_type=DEVICE_TYPE, model_id=MODEL_ID, model_basename=MODEL_BASENAME)
        st.session_state["LLM"] = LLM




    if "QA" not in st.session_state:

        prompt, memory = model_memory()

        QA = RetrievalQA.from_chain_type(llm=LLM, chain_type="stuff", retriever=RETRIEVER, return_source_documents=True,chain_type_kwargs={"prompt": prompt, "memory": memory},)
        st.session_state["QA"] = QA

def delete_source_route():
    folder_name = "SOURCE_DOCUMENTS"
    
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
      

    os.makedirs(folder_name)
    


def ingestdoc():
    if "result" in st.session_state:  
     del st.session_state["result"] 
     initialize_session_result_state()
     initialize_session_qa_state()
    else:   
        initialize_session_result_state()
        initialize_session_qa_state()
    
def ingestdocc():
   
        print("delete all keys  ")
        for key in st.session_state.keys():
            del st.session_state[key]
        print(st.session_state.keys())   
        initialize_session_result_state()
        initialize_session_qa_state()
        print("RAN")
       
# Sidebar contents

with st.sidebar:
    
    st.title(':red[_Converse with your Data_]')
    
    st.caption('Developed by Abhishek Vijra. AOT APAC RED HAT SG ') 

    st.caption('Powered by Red hat Openshift') 


    uploaded_files = st.file_uploader("Upload your Document", accept_multiple_files=True)
      
    if st.button('Delete Documents', help="click me to delete the documents you uploaded "):
      
          delete_source_route() 
          st.toast("Documents sucessfully Deleted.")
   
    if st.button('Create Brain', help="click me to create a context for AI with documents you uploaded "):
     with st.status("creating brain. please wait"):
          ingestdocc() 
          st.toast("New Brain Created. Now please start the Conversation with your Documents")

   
   
    
for uploaded_file in uploaded_files:
    string = uploaded_file.read()
    with open(os.path.join(SOURCE_DIRECTORY,uploaded_file.name),"wb") as f:
      f.write(uploaded_file.getbuffer())
    
   
    st.write("File name:", uploaded_file.name)
    st.success("Please click on Create Brain to create a AI context",icon="🤖")
    













    # Create a text input box for the user

with chat(key="my_chat"):
   
#prompt = st.text_input('Input your prompt here')
# while True:
    if prompt:= st.chat_input():
        add_message("user", prompt, avatar="🧑‍💻")
    # Then pass the prompt to the LLM
    
        response = st.session_state["QA"](prompt)
        answer, docs = response["result"], response["source_documents"]

        add_message("assistant", " AI: ", answer, avatar="🤖")
        #st.snow()
    # ...and write it out to the screen
    #st.write(answer)
    

    # With a streamlit expander  
#with st.expander('Relevant Data '):
        
        # Find the relevant pages
   # search = st.session_state.DB.similarity_search_with_score(prompt) 
        # Write out the first
   # for i, doc in enumerate(search): 
            # print(doc)
           # st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
           # st.write(doc[0].page_content) 
           # st.write("--------------------------------")