#!/usr/bin/env python
# coding: utf-8

# In[4]:


# !pip install streamlit streamlit-extras


# In[12]:


import torch
import subprocess
import streamlit as st
import logging
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY, MODEL_ID, MODEL_BASENAME
from langchain.embeddings import HuggingFaceInstructEmbeddings,HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
# from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain
from langchain.schema.document import Document
from langchain.storage import InMemoryStore

from run_inference import load_model,retrieval_qa_pipeline
from prompt_template_utils import get_prompt_template


# In[13]:


# setting up params:
device_type='cpu'
show_sources=True
use_history=True
save_qa=True
promptTemplate_type="llama"


# In[14]:


logging.info(f"Display Source Documents set to: {show_sources}")
print(f"Display Source Documents set to: {show_sources}")
logging.info(f"Display Use History set to: {use_history}")
print(f"Display Use History set to: {use_history}")
logging.info(f"Display promptTemplate_type set to: {promptTemplate_type}")
logging.info(f"Display Save QA set to: {save_qa}")


# In[16]:


QA,EMBEDDINGS,RETRIEVER,DB,LLM = retrieval_qa_pipeline(use_history, promptTemplate_type=promptTemplate_type,device_type=device_type)


# In[ ]:


# loading each objects into streamlit session states:

if "EMBEDDINGS" not in st.session_state:
    st.session_state.EMBEDDINGS = EMBEDDINGS

if "DB" not in st.session_state:
    st.session_state.DB = DB

if "RETRIEVER" not in st.session_state:
    st.session_state.RETRIEVER = RETRIEVER

if "LLM" not in st.session_state:
    st.session_state["LLM"] = LLM


if "QA" not in st.session_state:
    st.session_state["QA"] = QA



# In[ ]:


# Sidebar contents

with st.sidebar:
    st.title("🤗💬 I am a GenAI bot trained on Biology Concepts - plz ask me anything related?")
    st.markdown(
        """
    ## About
    Developed by Abhay Kumar for Quantiphi Interview Round.
    
    This is capable of answering questions on Biology Textbook -Chapter 1-2.
 
    """
    )
    #add_vertical_space(5)
    #st.write("Made by Abhay Kumar")
    
    
st.title("Biology_QA_bot 💬")

# Create a list to store prompt-answer pairs
prompt_history = []
# Create a text input box for the user
prompt = st.text_input("Input your prompt here")

# Create a submit button
submit_button = st.button("Submit")

# If the user hits enter
if submit_button:
    # Then pass the prompt to the LLM
    response = st.session_state["QA"](prompt)
    print("prompt sent to LLM for generation")
    answer, docs = response["result"], response["source_documents"]
    print("Answer ---->", answer)
#     st.write(answer)
    # ...and write it out to the screen
    # Store the prompt and its answer in the history
    prompt_history.insert(0, {"prompt": prompt, "answer": answer, "docs": docs})
    
    prompt=""
    # Add an empty line or some space to separate from the next prompt
    st.write("")  # or st.text("")
    
# Display the prompt history in reverse order
for entry in prompt_history:
    st.write(f"Prompt: {entry['prompt']}")
    st.write(f"Answer: {entry['answer']}")

    with st.expander("Document Similarity Search"):
        # Display relevant documents
        for i, doc in enumerate(entry["docs"]):
            st.write(f"Source Document # {i+1} : {doc.metadata['source'].split('/')[-1]}")
            st.write(doc.page_content)
            st.write("--------------------------------")
            
#         # Find the relevant pages
#         search = st.session_state.DB.similarity_search_with_score(prompt)
#         # Write out the first
#         for i, doc in enumerate(search):
#             # print(doc)
#             st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
#             st.write(doc[0].page_content)
#             st.write("--------------------------------")
    
    st.write("")  # Add an empty line to separate entries

    
#     st.write(answer)

#     # With a streamlit expander
#     with st.expander("Document Similarity Search"):
#         # Find the relevant pages
#         search = st.session_state.DB.similarity_search_with_score(prompt)
#         # Write out the first
#         for i, doc in enumerate(search):
#             # print(doc)
#             st.write(f"Source Document # {i+1} : {doc[0].metadata['source'].split('/')[-1]}")
#             st.write(doc[0].page_content)
#             st.write("--------------------------------")
    
#     st.write("") 
