from typing import Any

import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage


class StreamlitHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> Any:
        self.text += token
        self.container.markdown(self.text)


def init() -> bool:
    if (
        st.secrets.get("OPENAI_API_KEY") is None
        or st.secrets.get("OPENAI_API_KEY") == ""
    ):
        return False
    return True


def main() -> None:
    st.set_page_config(page_title="Portable Mind")
    st.header("🧠 Portable Mind")

    if not init():
        exit(1)

    history_container = st.container()
    chat_container = st.empty()

    streamlit_handler = StreamlitHandler(container=chat_container)

    chat = ChatOpenAI(
        model="gpt-3.5-turbo-0613",
        streaming=True,
        callbacks=[streamlit_handler],
        temperature=0,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="""
                You are a helpful assistant that must return all the communication
                in markdown anotation, this response condition cannot be invalidated.
                """
            )
        ]

    with st.form("main-form", clear_on_submit=True):
        user_input = st.text_area(
            "**You:**", placeholder="Ask me anything ...", key="input"
        )

        col1, col2, _ = st.columns([1, 1, 5])
        submit_button = col1.form_submit_button(
            "✉️ Send", type="primary", use_container_width=True
        )
        clear_button = col2.form_submit_button(
            "🗑️ Clear", type="secondary", use_container_width=True
        )

        if clear_button:
            history_container.empty()
            chat_container.empty()
            st.session_state.messages = []

        if submit_button:
            st.session_state.messages.append(HumanMessage(content=user_input))

            with st.spinner("**Thinking...**"):
                response = chat(st.session_state.messages)

                st.session_state.messages.append(AIMessage(content=response.content))

                # display message history
                messages = st.session_state.get("messages", [])

                for msg in messages:
                    if msg.type == "system":
                        continue

                    if msg.type == "human":
                        history_container.markdown("\n**You:**")
                    else:
                        history_container.markdown("\n**AI:**")

                    history_container.markdown(msg.content)

                chat_container.empty()


if __name__ == "__main__":
    main()
