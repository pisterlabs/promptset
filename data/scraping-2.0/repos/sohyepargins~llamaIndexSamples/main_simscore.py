from dotenv import load_dotenv
import os
load_dotenv()
import pinecone
from llama_index import (
    download_loader,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext,
    VectorStoreIndex    
)
from llama_index.llms import OpenAI
from llama_index.vector_stores import PineconeVectorStore 

from llama_index.callbacks import LlamaDebugHandler, CallbackManager

import streamlit as st



#To run in Terminal, run in the terminal %streamlit run main.py to check

print("***Streamlit LlamaIndex Documentation Helper")

@st.cache_resource(show_spinner=False) 

def get_index()-> VectorStoreIndex:  
    
    pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENV"])  

    pinecone_index = pinecone.Index(index_name="llamaindex-documentation-helper")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    #Add callbacks to the ServiceContext
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager= CallbackManager(handlers=[llama_debug])
    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)  
        
    return  VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

index = get_index() 
        
st.set_page_config(page_title="Chat with LlamaIndex docs powered by llamaIndex ",
                       page_icon="🦙",
                       layout = "centered",
                       initial_sidebar_state="auto",
                       menu_items=None
                       )
st.title("Chat with LlamaIndex docs 🦙")

if "chat_engine" not in st.session_state.keys():
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="react", verbose = True)

    
if "messages" not in st.session_state.keys():
    st.session_state.messages=[
        {
            "role": "assistant",
            "content": "Ask me a question about LlamaIndex's open source python library." 
        }
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
  
for message in st.session_state.messages:
    with st.chat_message(message["role"]): 
        st.write(message["content"])
        
if st.session_state.messages[-1]["role"] != "assistant": 
    with st.chat_message("assistant"): 
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(message=prompt) 
            st.write(response.response) 
            nodes = [node for node in response.source_nodes]
            for col, node, i in zip(st.columns(len(nodes)), nodes, range(len(nodes))):
                with col:
                    st.header(f"Source Node {i+1}: score = {node.score}")
                    st.write(node.text)
            message = {        
                "role": "assistant",
                "content":response.response
            }
            st.session_state.messages.append(message)

