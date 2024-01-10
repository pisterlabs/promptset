from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="FinChat",
    page_icon="🧠",
)
st.sidebar.title("LangChain")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if "messgaes" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "hello, I'm the assistant. I can answer questions about the CSV file and the web.",
        }
    ]


for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])


if prompt := st.chat_input(placeholder="Type something..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()

    llm = ChatOpenAI(
        temperature=0, streaming=True, openai_api_key=openai_api_key
    )
    # tools = load_tools(["ddg-search"])
    searchagent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = searchagent.run(st.session_state.messages, callbacks=[st_callback])
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.write(response)
