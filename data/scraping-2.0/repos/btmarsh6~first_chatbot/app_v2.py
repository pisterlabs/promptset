import streamlit as st
import sqlite3
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory


# App Title and Page Config
st.set_page_config(page_title='SQL Chatbot')
with st.sidebar:
    st.title('SQL Chatbot')
    st.write("Use me to help you explore your database!")
    
# OpenAI Credentials
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()

conn = sqlite3.connect('chatbot_database.db')


# Create the agent executor
db = SQLDatabase.from_uri("sqlite:///./chatbot_database.db")
toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0,
                                               openai_api_key=openai_api_key))
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(chat_memory=msgs)

agent_executor = create_sql_agent(
    llm=OpenAI(temperature=0, openai_api_key=openai_api_key),
    toolkit=toolkit,
    verbose=True,
    agent_executor_kwargs={'memory': memory}
)

view_messages = st.expander("View the message contents in session state")

# Opening message
if len(msgs.messages) == 0:
    with st.chat_message("ai"):
        st.write("How may I assist you?")

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    with st.chat_message("human"):
        st.write(prompt)
    response = agent_executor.run(prompt)
    with st.chat_message("ai"):
        st.write(response)


with view_messages:
    """
    Memory initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    memory = ConversationBufferMemory(chat_memory=msgs)
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)


def clear_chat_history():
    msgs.clear()
    st.session_state.messages = [{"role": "ai", "content": "How may I assist you today?"}]


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
