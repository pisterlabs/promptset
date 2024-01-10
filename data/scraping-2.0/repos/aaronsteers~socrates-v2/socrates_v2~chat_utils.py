import json
import os
from typing import TYPE_CHECKING

import streamlit as st

from socrates_v2.interop import LangChainStreamingCallbackToStreamlit
from socrates_v2.ai import new_personality, PERSONALITIES

if TYPE_CHECKING:
    from socrates_v2.ai import AIPersonality

def new_chat(personality_name="Socrates", history: list[dict] | None = None):
    """Start a new chat with the given personality.

    This resets all prior messages and langchain artifacts.
    """
    # Initialize the personality:
    personality: AIPersonality = new_personality(personality_name)
    st.session_state["personality"] = personality
    st.session_state["chain"] = personality.create_langchain_chain(
        openai_api_key=st.session_state.openai_api_key
    )
    st.session_state["messages"] = history or []


def show_sidebar():
    with st.sidebar:
        st.text_input(
            label="OpenAI API Key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            key="openai_api_key",
        )
        st.text("Select your sage:")
        st.button(
            "New conversation with Socrates",
            key="new_socrates",
            on_click=new_chat,
            args=("Socrates",),
            kwargs={},
        )
        st.button(
            "New conversation with Yoda",
            key="new_yoda",
            on_click=new_chat,
            args=("Yoda",),
            kwargs={},
        )
        if st.checkbox("Enable debug mode"):
            show_debug_info()


def show_debug_info():
    # format dict as json with indentation of 2 spaces
    with st.sidebar:
        st.subheader("History:")
        with st.container():
            if "messages" in st.session_state:
                json_msg_hist = json.dumps(st.session_state.messages, indent=2)
                st.code(json_msg_hist, language="json")
            else:
                st.warning("No chat history yet.")


def show_chat_history():
    if "messages" not in st.session_state or not st.session_state["messages"]:
        st.session_state["messages"] = [
            {
                "role": st.session_state.personality.name,
                "content": st.session_state.personality.get_greeting()
            }
        ]

    for msg in st.session_state["messages"]:
        avatar = None  # default to a generic user avatar
        if msg["role"] in PERSONALITIES:
            avatar = PERSONALITIES[msg["role"]].avatar

        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])


def show_chat():
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.error(
            "Please add your OpenAI API key in the sidebar to continue.\n\n" +
            "You can use the following link to generate a new key: " +
            "https://platform.openai.com/account/api-keys"
        )
        return

    if "personality" not in st.session_state:
        # We don't know who we are speaking with yet.
        return

    show_chat_history()

    user_query = st.chat_input(placeholder="")
    if user_query:
        # Immediately echo the input into chat and append to history:
        st.chat_message("user").write(user_query)
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )

        # Then append (stream) the response from the chatbot:
        append_ineractive_chat(user_query)


def append_ineractive_chat(user_query: str):
    personality = st.session_state.personality
    chain = st.session_state.chain

    with st.chat_message(
        st.session_state.personality.name,
        avatar=personality.avatar,
    ):
        st_typing_region = st.empty()  # New empty container for the typing effect.
        response = chain.run(
            user_query,
            callbacks=[
                LangChainStreamingCallbackToStreamlit(st_typing_region)
            ]
        )

        # The response is streamed in realtime, above. Once we reach here, we
        # can add the full response to the chat history:
        st.session_state.messages.append(
            {"role": st.session_state.personality.name, "content": response}
        )
