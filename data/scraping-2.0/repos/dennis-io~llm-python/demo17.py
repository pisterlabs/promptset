import os
from dotenv import load_dotenv

from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
import streamlit as st

load_dotenv()

st.set_page_config(page_title="LangChain Agents + MKRL", page_icon="🤖")
st.title("🤖 LangChain Agents + MKRL")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "message": "Hello, I'm the assistant. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["message"])

if prompt := st.chat_input(placeholder="Who won Hardrock 100?"):
    st.session_state.messages.append({"role": "user", "message": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please enter your OpenAI API Key")
        st.stop()

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True
    )
    search_agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
    )
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "message": response})
        st.write(response)
