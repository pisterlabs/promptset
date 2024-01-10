import streamlit as st
from langchain.memory import XataChatMessageHistory


def xata_chat_history(_session_id: str):
    chat_history = XataChatMessageHistory(
        session_id=_session_id,
        api_key=st.secrets["xata_api_key"],
        db_url=st.secrets["xata_db_url"],
        table_name="tiangong_memory",
    )

    return chat_history
