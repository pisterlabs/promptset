from abc import ABCMeta
from langchain_services. import Messages

class MessageInterface(metaclass=ABCMeta):
    def __init__(self):
        self.messages = Messages()

    def parse_message(self, message):
        """
        Parses a message into a message object.

        Args:
            message (str): The message to parse.

        Returns:
            obj: The parsed message.
        """
        self.message = message
        self.id = self.message["id"]
        self.role = self.message["role"]
        self.content = self.message["content"]
        self.timestamp = self.message["timestamp"]
        return self.message
