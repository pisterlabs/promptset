import time

import openai
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents.agent_toolkits import VectorStoreInfo, VectorStoreToolkit
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.vectorstores import Chroma

from create_index import CHROMA_PERSIST_DIRECTORY

ENABLE_STREAMING_DELAY = True
STREAMING_DELAY_SEC = 0.01

openai.log = "debug"

load_dotenv()


def create_agent(model_name, memory):
    llm = ChatOpenAI(model_name=model_name, temperature=0, streaming=True)

    # Setup DuckDuckGo and Wikipedia
    tools = load_tools(["ddg-search", "wikipedia"])

    # Setup VectorStore
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        embedding_function=embeddings, persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    vectorstore_info = VectorStoreInfo(
        vectorstore=db,
        name="langchain-streamlit-example",
        description="Source code of application named `langchain-streamlit-example`",
    )
    vectorstore_toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info, llm=llm)
    vectorstore_tools = vectorstore_toolkit.get_tools()
    tools.extend(vectorstore_tools)

    # Setup Memory
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )


st.title("langchain-streamlit-example")

model_name = st.radio("Model", ("gpt-3.5-turbo", "gpt-4"), horizontal=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="memory", return_messages=True
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callbacks = []

        st_callback = StreamlitCallbackHandler(st.container())
        callbacks.append(st_callback)

        if ENABLE_STREAMING_DELAY:
            # Streamingで動いていることが分かりやすいようにするためのコールバック
            class DelayCallbackHandler(BaseCallbackHandler):
                def on_llm_new_token(self, token: str, **kwargs) -> None:
                    time.sleep(STREAMING_DELAY_SEC)

            delay_callback = DelayCallbackHandler()
            callbacks.append(delay_callback)

        # モデルが毎回変更されるかもしれないため、Agentは毎回作成することになる
        agent = create_agent(model_name, st.session_state.memory)
        response = agent.run(prompt, callbacks=callbacks)
        st.markdown(response)
        st.write(f"model: {model_name}")

    st.session_state.messages.append({"role": "assistant", "content": response})
