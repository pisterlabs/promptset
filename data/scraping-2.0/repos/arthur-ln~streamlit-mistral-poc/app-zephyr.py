# app.py

from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from dotenv import load_dotenv
from typing import List, Union
import streamlit as st
import openai
import os
import chromadb
from transformers import AutoTokenizer

def init_page():
    st.set_page_config(
        page_title="Jarvis Chatbot - Demo Zephyr X Garde Nationale",
    )
    st.header("Jarvis - Demo Zephyr X Garde Nationale")
    st.sidebar.title("Options")
    st.sidebar.checkbox("Utiliser les contextes", key="use-context", on_change=use_context_button)
    st.sidebar.checkbox("Afficher les contextes", key="display-context", on_change=display_context_button)

def init_messages():
    clear_button = st.sidebar.button("Effacer la conversation", key="clear")
    
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="Your name is Jarvis, you are a helpful AI assistant. Reply your answer in French.")
        ]
        st.session_state.contexts = []
        st.session_state.use_ctx_checked = False
        st.session_state.display_ctx_checked = False

def use_context_button():
    st.session_state.use_ctx_checked = not st.session_state.use_ctx_checked

def display_context_button():
    st.session_state.display_ctx_checked = not st.session_state.display_ctx_checked


def get_answer(user_input, tokenizer, contexts):
    
    if st.session_state.use_ctx_checked:
        messages = add_contexts_to_prompt(user_input, contexts)
        print('context added')
    else :
        messages = convert_langchainschema_to_dict(st.session_state.messages)
        print('no context added')
    print(messages)
        
    messages_w_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    chat_completion = openai.ChatCompletion.create(model="HuggingFaceH4/zephyr-7b-beta",
    temperature=0.1,
    max_tokens=1024,
    messages=messages_w_template)
    
    response = chat_completion['choices'][0]['message']['content']
    return response


def convert_langchainschema_to_dict(messages: List[Union[SystemMessage, HumanMessage, AIMessage]]):
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def find_role(message):
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")


def get_contexts(query: str):
    # TODO : setup un système d'authentification
    chroma_client = chromadb.HttpClient(
        host="57.128.17.86",
        port="8000",
    )
    # TODO : rajouter la fonction d'embedding en argument
    context_collection = chroma_client.get_or_create_collection("contexts")
    results = context_collection.query(query_texts=[query], n_results=10)
    return results["documents"][0]


def add_contexts_to_prompt(user_input, context):
    #we add some context to answer the question
    messages = convert_langchainschema_to_dict(st.session_state.messages)
    user_input_prompted = f"With the following contexts : {context} ; answer the following question : {user_input}"
    messages[-1]['content'] = user_input_prompted
    return messages
    
    
def main():
    init_page()
    init_messages()
    
    #initialization fo the tokenizer
    checkpoint = "HuggingFaceH4/zephyr-7b-beta"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    #connexion to the GCP VM
    load_dotenv()
    openai.api_base = os.getenv('DOMAIN_NAME')  # use the IP or hostname of your instance
    openai.api_key = "none"  # vLLM server is not authenticated

    # Supervise user input
    if user_input := st.chat_input("Rentrez votre question !"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        #we get the context from the database
        if st.session_state.use_ctx_checked:
            contexts = get_contexts(user_input)
        else :
            contexts = ""
        st.session_state.contexts.append(contexts)
        
        with st.spinner("Zephyr est en train d'écrire ..."):
            answer = get_answer(user_input, tokenizer, contexts)
        st.session_state.messages.append(AIMessage(content=answer))

    # Display chat history
    messages = st.session_state.get("messages", [])
    contexts = st.session_state.get("contexts", [])
    index = 0
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
                if st.session_state.display_ctx_checked and contexts[index] != "":
                    st.markdown(f'<p style="color:Green; font-size: 15px;"> Les contextes utilisés depuis le site de la garde nationale sont les suivants : <br>{contexts[index]}</br> </p>', unsafe_allow_html=True)
                index+=1
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


# streamlit run app.py
if __name__ == "__main__":
    main()