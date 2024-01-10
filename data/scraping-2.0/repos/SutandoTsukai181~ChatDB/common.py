import re
from datetime import datetime
from typing import Dict, List, Tuple

import openai
import streamlit as st


class DatabaseProps:
    id: str
    uri: str

    def __init__(self, id, uri) -> None:
        self.id = id
        self.uri = uri

    def get_uri_without_password(self) -> str:
        match = re.search("(:(?!\/\/).+@)", self.uri)

        if not match:
            return self.uri

        # Use fixed password length
        return f'{self.uri[:match.start(0) + 1]}{"*" * 8}{self.uri[match.end(0) - 1:]}'


class Message:
    role: str
    content: str

    query_results: List[Tuple[str, list]]

    def __init__(self, role, content, query_results=None) -> None:
        self.role = role
        self.content = content

        self.query_results = query_results or []


class Conversation:
    id: str

    agent_model: str

    database_ids: List[str]

    messages: List[Message]
    query_results_queue: List[Tuple[str, str, list]]

    # Used to invalidate get_agent() cache
    # Whenever we update the database ids of a conversation, we update this timestamp
    # so that get_agent() will be re-executed
    last_update_timestamp: float

    def __init__(
        self,
        id: str,
        agent_model: str,
        database_ids: List[str],
        messages: List[Message] = None,
    ) -> None:
        self.id = id
        self.agent_model = agent_model

        self.database_ids = list(database_ids)

        self.messages = list(messages) if messages else list()
        self.query_results_queue = list()

        self.update_timestamp()

    def add_message(self, role, content, query_results=None):
        self.messages.append(Message(role, content, query_results))

    def update_timestamp(self):
        self.last_update_timestamp = datetime.now().timestamp()


def init_session_state():
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""

    if "databases" not in st.session_state:
        st.session_state.databases: Dict[str, DatabaseProps] = dict()

    if "conversations" not in st.session_state:
        st.session_state.conversations: Dict[str, Conversation] = dict()

    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation: str = ""

    if "retry" not in st.session_state:
        st.session_state.retry = None


def set_openai_api_key(api_key):
    # Set API key in openai module
    openai.api_key = api_key
    st.session_state.openai_key = api_key
