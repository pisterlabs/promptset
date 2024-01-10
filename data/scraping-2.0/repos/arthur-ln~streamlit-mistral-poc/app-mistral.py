# app.py

from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from dotenv import load_dotenv
from typing import List, Union
import streamlit as st
import openai
import os
import chromadb
#from fastchat.conversation import get_conv_template

def init_page():
    st.set_page_config(
        page_title="Jarvis Chatbot - Demo Mistral X Garde Nationale",
    )
    st.header("Jarvis - Demo Mistral X Garde Nationale")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Effacer la conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Reply your answer in mardkown format.")
        ]
        st.session_state.costs = []

def get_answer(messages):
    chat_completion = openai.ChatCompletion.create(model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.1,
    max_tokens=1024,
    messages=messages)
    response = chat_completion['choices'][0]['message']['content']
    return response


def convert_langchainschema_to_dict(messages: List[Union[SystemMessage, HumanMessage, AIMessage]]):
    """
    Convert the chain of chat messages in list of langchain.schema format to
    list of dictionary format.
    """
    return [{"role": find_role(message),
             "content": message.content
             } for message in messages]


def find_role(message):
    """
    Identify role name from langchain.schema object.
    """
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    raise TypeError("Unknown message type.")

# def add_prompt_template(user_input):
#     user_input_prompted = f"<s>[INST] {user_input} [/INST] Model answer</s>[INST] Réponds en français de manière concise[/INST]"
#     return user_input_prompted

# def add_conversational_template(messages):
#     template = get_conv_template("mistral")
#     template.messages = messages
#     print('template', template)
#     return template

def get_contexts(query: str):
    # TODO : setup un système d'authentification
    chroma_client = chromadb.HttpClient(
        host="57.128.17.86",
        port="8000",
    )
    # TODO : rajouter la fonction d'embedding en argument
    context_collection = chroma_client.get_or_create_collection("contexts")
    results = context_collection.query(query_texts=[query], n_results=10)
    # print(context_collection.peek()["documents"])
    # print(context_collection.count())
    return results["documents"]

def add_contexts_to_prompt(user_input, context):
    #we add some context to the template prompt
    user_input_prompted = f"<s>[INST] With the following contexts : {context}, answer the following question : {user_input} [/INST] Model answer</s>[INST] Réponds en français de manière claire [/INST]"
    return user_input_prompted
    


def main():
    init_page()
    init_messages()
    
    load_dotenv() #specific for development phase
    
    openai.api_base = os.getenv('DOMAIN_NAME')  # use the IP or hostname of your instance
    openai.api_key = "none"  # vLLM server is not authenticated

    # Supervise user input
    if user_input := st.chat_input("Rentrez votre question !"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        #we get the context from the database
        
        contexts = get_contexts(user_input)
        print('contexts from chromadb \n', contexts)
        
        with st.spinner("Mistral est en train d'écrire ..."):
            #messages = convert_langchainschema_to_dict(st.session_state.messages)
            user_input_w_context = add_contexts_to_prompt(user_input, contexts)
            answer = get_answer(user_input_w_context)
        st.session_state.messages.append(AIMessage(content=answer))

    # Display chat history
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


# streamlit run app.py
if __name__ == "__main__":
    main()