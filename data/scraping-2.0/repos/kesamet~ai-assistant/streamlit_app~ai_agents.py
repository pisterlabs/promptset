import streamlit as st
from langchain.schema import HumanMessage, AIMessage

from src.agents import build_agent


def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="ai_agents")
    if clear_button or "aa_messages" not in st.session_state:
        st.session_state.aa_messages = []


def get_output(user_input: str, messages: list) -> str:
    agent = build_agent(messages)
    try:
        return agent.run(user_input)
    except Exception:
        return "GoogleGenerativeAI is not available. Did you provide an API key?"


def ai_agents():
    st.sidebar.title("AI Agents")
    st.sidebar.info(
        "AI Assistant is powered by text-bison and has access to wikipedia, search, "
        "News API, Wolfram and calculator tools."
    )

    init_messages()

    # Display chat history
    for message in st.session_state.aa_messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)

    if user_input := st.chat_input("Your input"):
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking ..."):
                output = get_output(user_input, st.session_state.aa_messages)
            st.markdown(output)

        st.session_state.aa_messages.append(HumanMessage(content=user_input))
        st.session_state.aa_messages.append(AIMessage(content=output))
