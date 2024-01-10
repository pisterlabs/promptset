""" This module represents chat messages."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Union

import streamlit as st
from ansi2html import Ansi2HTMLConverter
from chatbot.catalog import SIMPLE_CHATBOT, UPLOAD_DOCUMENT_SEARCH, ModelCatalogItem, RetrieverCatalogItem, FlowCatalogItem


class ChatParticipant(Enum):
    """Participants that are able to send messages to the chat."""

    USER = "user"
    BOT = "assistant"
    DEBUGGER = "debugger"


@dataclass
class AbstractChatMessage(ABC):
    """Represents a chat message."""

    @property
    @abstractmethod
    def sender(self) -> ChatParticipant:
        """The sender of this message."""

    @property
    def avatar(self) -> Union[str, None]:
        """Path to image or name of icon to use as representation of the sender of this message.
        See also https://docs.streamlit.io/library/api-reference/chat/st.chat_message.
        """
        return None

    @abstractmethod
    def write(self):
        """Displays this message using Streamlit elements."""


@dataclass
class ChatMessage(AbstractChatMessage):
    """A chat message is a regular message sent to the chat."""

    _sender: ChatParticipant
    _msg: str

    @property
    def sender(self) -> ChatParticipant:
        """See base class."""
        return self._sender

    def write(self):
        """See base class."""
        st.write(self._msg, unsafe_allow_html=True)


@dataclass
class InfoMessage(AbstractChatMessage):
    """An info message displays information in a chat
    that has not be authored by the chatbot."""

    _sender: ChatParticipant
    _msg: str

    @property
    def sender(self) -> ChatParticipant:
        """See base class."""
        return self._sender

    def write(self):
        """See base class."""
        st.info(self._msg)


@dataclass
class DebugMessage(AbstractChatMessage):
    """A debug message gives additional transparency about
    what the chatbot does in the background."""

    def __init__(self, sender: ChatParticipant, msg: str):
        # Get the label for expander from the first line
        # and generate the message with the rest
        lines = msg.split("\n")
        self.label = lines[0]
        # Convert new lines to markdown new lines (two blank spaces)...
        self._msg = "  \n".join(lines[1:])
        # ... and ANSI from LangChain to HTML
        converter = Ansi2HTMLConverter()
        self._sender = sender
        self._msg = converter.convert(self._msg)

    @property
    def sender(self) -> ChatParticipant:
        """See base class."""
        return self._sender

    @property
    def avatar(self) -> str:
        """See base class."""
        # return "src/icons/llm-xray-chat.png"
        return "src/icons/X-Ray.png"

    def write(self):
        """See base class."""
        expander = st.expander(label=self.label, expanded=False)
        expander.markdown(self._msg, unsafe_allow_html=True)


class ChatHistory:
    """Stateful UI component that stores chat messages."""

    def __init__(
        self, chatbot_name: str, gettext: Callable[[str], str] = lambda x: x
    ) -> None:
        super().__init__()
        self.chatbot_name = chatbot_name
        self.gettext = gettext

        if "chat_history" not in st.session_state:
            self.add_chat_message(
                ChatMessage(ChatParticipant.BOT, self.how_may_i_help_you_msg())
            )

    def how_may_i_help_you_msg(self) -> str:
        """The message text that the chatbot introduces itself with.

        Returns:
            A string that contains the welcome message.
        """
        _ = self.gettext
        return _("Hi, I'm {chatbot_name}. How may I help you?").format(
            chatbot_name=self.chatbot_name
        )

    def create_system_message(
        self, model: ModelCatalogItem, retriever: RetrieverCatalogItem, flow: FlowCatalogItem
    ) -> InfoMessage:
        """Creates a system message without storing it in the chat history.
            A system message is informs the human about the LLM and information
            source that the chatbot currently uses.

        Args:
            model: The current model that the chatbot uses.
            retriever: The current document source that the chatbot uses.
            flow: The current flow that the chatbot uses

        Returns:
            A system message that informs about the flow, document source and LLM.
        """
        _ = self.gettext
        flow_name = flow.friendly_name # can't be undefined!
        flow_search_msg = (
                _("The chosen flow is {flow_name}.").format(
                    flow_name=flow_name
                )
            )
        retriever_name = retriever.friendly_name if retriever is not None else "No Retriever"

        document_search_msg = (
            ""
            if retriever_name == SIMPLE_CHATBOT or retriever_name == UPLOAD_DOCUMENT_SEARCH
            else " "
            + _("The information source is {retriever_name}.").format(
                retriever_name=retriever_name
            )
        )

        if model:
            model_msg = _("You are chatting with {model_name}.").format(
                model_name=model.friendly_name
            )
        else:
            model_msg = ""
        
        return InfoMessage(ChatParticipant.BOT, model_msg + document_search_msg + " " + flow_search_msg)

    def clear(self):
        """Resets the chat history to welcome message from bot."""
        del st.session_state["chat_history"]
        self.add_chat_message(
            ChatMessage(ChatParticipant.BOT, self.how_may_i_help_you_msg())
        )

    def add_log_message(self, msg):
        # TODO
        if "log_history" not in st.session_state:
            st.session_state["log_history"] = []
            return
        len(st.session_state.log_history)

    def add_chat_message(self, msg: AbstractChatMessage):
        """Stores a chat message in this chat history.

        Args:
            msg: The message to add to the chat history.
        """

        # chat_history is a list of lists of messages.
        # it is a list of lists so that a chat participant can have multiple messages in a row
        # without showing the sender information multiple times.
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [
                {
                    "sender": msg.sender,
                    "messages": [msg],
                    "avatar": msg.avatar,
                }
            ]
            return
        num_chat_msgs = len(st.session_state.chat_history)
        if num_chat_msgs > 0:
            previous = st.session_state.chat_history[-1]
            if previous["sender"].value == msg.sender.value:
                st.session_state.chat_history[-1]["messages"].append(msg)
                return

        st.session_state.chat_history.append(
            {"sender": msg.sender, "messages": [msg], "avatar": msg.avatar}
        )

    def write(self):
        """Display this chat history using Streamlit elements."""
        if "chat_history" in st.session_state:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["sender"].value, avatar=message["avatar"]):
                    for msg in message["messages"]:
                        msg.write()
