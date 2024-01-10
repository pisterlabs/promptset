import streamlit as st
import random
import time
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
os.environ['OPENAI_API_KEY'] = "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"
os.environ['SERPAPI_API_KEY'] = '360d22e4bc0b06f384cdc79db107bd5ef547daa1c1843698dfcff447654b98e5'

st.title("chatbot for coding")
@st.cache_resource(ttl=10800) 
def create_conversation_chain():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "The following is conversation between a coder and an AI expert in codeing. The AI "
            "provides lots of specific details from its context. If the AI does not know the answer to a "
            "question, it truthfully says it does not know."
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    return conversation

conversation = create_conversation_chain()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("please enter content?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Display assistant response in chat message container

    
    assistant_response = conversation.predict(input=prompt)
    if assistant_response:
        with st.chat_message("assistant"):
            # message_placeholder = st.empty()
            # full_response = ""
            # # Simulate stream of response with milliseconds delay
            # for chunk in assistant_response.split():
            #     full_response += chunk + " "
            #     time.sleep(0.05)
            #     # Add a blinking cursor to simulate typing
            #     message_placeholder.markdown(full_response + "▌")
            # message_placeholder.markdown(full_response)
            st.write(assistant_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})