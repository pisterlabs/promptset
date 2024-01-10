import datetime
import os
from nova_chat.constants import IO_DIR, RemoteLLM, APP_USERS
from nova_chat.llm_client import LLMFactory

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
from icecream import ic
import pandas as pd
from nova_chat.io import (
    save_message,
    load_message,
    delete_message,
)

def getConversation(memory, model, st=None):
    chat = LLMFactory.getChat(model, st)
    
    # Prompt 
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You are a nice chatbot having a conversation with a human."
            ),
            # The `variable_name` here is what must align with memory
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    return LLMChain(
            llm=chat,
            prompt=prompt,
            verbose=False,
            memory=memory
        )

def populate_messages_to_memory(messages, memory):
    """Build up the prompt memory from the session state messages."""
    input, output = None, None
    for message in messages:
        if message["role"] == "user":
            input = message["content"]
        else:
            output = message["content"]
        
        if input and output:
            memory.save_context({"input": input}, {"output": output})
            input, output = None, None
            
def build_model_loader_sidebar():
    with st.sidebar:
        with st.container():
            v = st.selectbox(
                "What model do you want to use?",
                (x.value.label for x in RemoteLLM),
            )
            model = RemoteLLM.get_enum_by_model(v)
            st.markdown("#")
    return model

def build_chat_io_sidebar():
    """Build the chat io sidebar."""
    with st.sidebar:
        with st.container():
            user = st.radio("Select user", APP_USERS)
            project_dir = os.path.join(IO_DIR, user)
            filename = st.text_input("Save conversation history to file","test")
            filename = os.path.join(project_dir, filename)
            
            save_col, clear_col = st.columns(2)
            with save_col:
                if st.button("Save chat",type="primary"):
                    if st.session_state.messages:
                        p = save_message(filename, st.session_state.messages)
                        st.success(f"Saved to {p}")
                    else:
                        st.error("Empty session state!")
            with clear_col:
                if st.button('Clear chat'):
                    st.session_state.messages = []
            
            with st.expander("List saved conversations:"):
                if not os.path.exists(project_dir):
                    os.makedirs(project_dir)
                    
                files = os.listdir(project_dir)
                    
                file_modified_time = [
                    datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(project_dir, file))).date() for file
                    in files
                ]
                file_sizes = [
                    format(os.path.getsize(os.path.join(project_dir, file)) / 1024, f".2f") for
                    file in files]
                files_with_time = pd.DataFrame(data=[files, file_modified_time, file_sizes],
                    index=['Model', 'Last modified', 'Size in KB']).T
                st.dataframe(files_with_time.set_index("Model"))
                
                file = st.selectbox("Select a memory file", files)
                if file:
                    file = os.path.join(project_dir, file)
                
                load_col, delete_col = st.columns(2)
                with load_col:
                    if st.button("Load",type="primary") and file:
                        messages = load_message(file)
                        st.session_state.messages = messages
                        st.success("Loaded!")
                with delete_col:
                    if st.button("Delete") and file:
                        messages = delete_message(file)
                        st.success("Deleted!")
        st.markdown("#")


def build_streamlit_demo():
    
    model = build_model_loader_sidebar()
    memory = ConversationBufferWindowMemory(k=30, memory_key="chat_history", return_messages=True)
    
    build_chat_io_sidebar()
                    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input("What's up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        populate_messages_to_memory(st.session_state.messages, memory)
        chat = getConversation(memory,model, st)
        res = chat({"question":prompt})
        st.session_state.messages.append({"role": "assistant", "content": res["text"]})