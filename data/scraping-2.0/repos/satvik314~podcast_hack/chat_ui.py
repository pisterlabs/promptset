import streamlit as st
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from utils import insert_into_db, embed_text, create_prompt
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


def chat_interface():
        # Initialize session state variables
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["I am Nexus! You can chat with me about your podcasts."]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    # Initialize ChatOpenAI and ConversationChain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    system_prompt = """You are bot which answers questions based on podcasts. You give short and conversational answers."""

    prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                MessagesPlaceholder(variable_name = "history"),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
    
    conversation = ConversationChain(prompt=prompt, llm=llm, memory=st.session_state.buffer_memory)

    response_container = st.container()
    spinner_container = st.container()
    text_container = st.container()

    with text_container:
        query = st.text_input("Query: ", key="input")

    with spinner_container:
        if query:
            with st.spinner("typing..."):
                # response = conversation.predict(input=query)
                response = conversation.predict(input=create_prompt(query))
                print(create_prompt(query))
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)


    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')