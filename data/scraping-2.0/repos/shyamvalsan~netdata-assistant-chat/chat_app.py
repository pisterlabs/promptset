import streamlit as st
from streamlit_feedback import streamlit_feedback
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
import os
import sys
import base64

filename_fn = lambda filename: {'file_name': filename}

st.set_page_config(page_title="Netdata Assistant", page_icon="logo.png", layout="wide", initial_sidebar_state="collapsed", menu_items=None)

openai_model = "gpt-3.5-turbo"

openai.api_key = st.secrets.openai_key
st.title("⚡Netdata Assistant")

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Netdata!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the latest Netdata docs – hang tight! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader('./data_v4/', recursive=True, file_metadata=filename_fn).load_data()
        #print([x.metadata['file_name'] for x in documents])
        service_context = ServiceContext.from_defaults(llm=OpenAI(model=openai_model, temperature=0.5, system_prompt="You are Netdata Assistant, an expert in all things related to Netdata. Respond in an intelligent and professional manner with clear and detailed answers. Answer based on facts – do not hallucinate. If you do not know the answer, point the user to https://community.netdata.cloud/"))
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        return index
    
index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=False)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    sys.stdout.write(f"Question: {prompt}\n")

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            file_name = response.source_nodes[0].node.metadata['file_name']
            print(file_name)
            url = None

            if "data_v4/docs" in file_name:
                with open(file_name, 'r') as file:
                    content = file.read()
                    for line in content.splitlines():
                        #print(line)
                        if line.startswith("learn_link:"):
                            url = line.split(" ")[1].strip()
                            break

            st.write(response.response)
            if url:  # Check if url is not None before writing it
                st.write("Read more: ",url)
                #st.markdown(f"[Read more]({url})")
            message = {"role": "assistant", "content": response.response}
            feedback = streamlit_feedback(feedback_type="thumbs", align="flex-start")
            st.session_state.messages.append(message) # Add response to message history 
            sys.stdout.write(f"Response: {message['content']}\n")
