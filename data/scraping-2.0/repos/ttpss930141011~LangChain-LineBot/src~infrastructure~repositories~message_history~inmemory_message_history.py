from typing import Dict, List, TypeVar

from langchain.schema import BaseChatMessageHistory
from langchain.schema.messages import BaseMessage

KT = TypeVar("KT", bound=str)
VT = TypeVar("VT", bound=List[BaseMessage])


class InMemoryChatDictMessageHistory(BaseChatMessageHistory):
    """In memory implementation of chat message history.

    Stores messages in an in memory list.
    """

    def __init__(self, window_id: str, data: Dict[KT, VT]) -> None:
        self.window_id = window_id
        self.data = data

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Retrieve the messages from Dict"""
        if self.window_id not in self.data:
            self.data[self.window_id] = []
        return self.data[self.window_id]

    def add_message(self, message: BaseMessage) -> None:
        """Add a self-created message to the store"""
        if self.window_id not in self.data:
            self.data[self.window_id] = [message]
        else:
            self.data[self.window_id].append(message)

    def clear(self) -> None:
        """
        Clears the data associated with the current window ID.

        Parameters:
            None

        Returns:
            None
        """
        self.data[self.window_id] = []
