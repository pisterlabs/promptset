"""Class for one message that is either a prompt or a completion."""
import uuid

import yaml
from openai.types.chat.chat_completion import ChatCompletion


class Message:
    """
    Class representing a message.

    The message can be either a user prompt or an AI completion.
    """

    @staticmethod
    def from_prompt(content: str, role: str = "user"):
        """
        Create a Message instance from a user prompt.

        Args:
            content (str): The text of the user prompt.
            role (str, optional): The role of the message. Defaults to 'user'.

        Returns:
            Message: The created Message instance.
        """
        return Message("chatprmt-" + str(uuid.uuid4()), role, content)

    @staticmethod
    def from_completion(completion: ChatCompletion):
        """
        Create a Message instance from an AI completion.

        Args:
            completion (ChatCompletion): The completion object from
                the OpenAI API.

        Returns:
            Message: The created Message instance.
        """
        message = completion.choices[0].message
        return Message(completion.id, message.role, message.content)

    def __init__(self, message_id: str, role: str, content: str):
        """
        Initialize a Message instance with the specified attributes.

        Args:
            message_id (str): Unique identifier for the message.
            role (str): The role of the message.
            content (str): The content of the message.
        """
        self._message_id = message_id
        self._role = role
        self._text = content

    @property
    def role(self) -> str:
        """
        Return the role of the user.

        Returns:
            str: The role of the user.
        """
        return self._role

    @property
    def content(self) -> str:
        """
        Return the content of the message.

        Returns:
            str: The content of the message.
        """
        return self._text

    @property
    def message_id(self) -> str:
        """
        Return the unique identifier of the message.

        Returns:
            str: The unique identifier of the message.
        """
        return self._message_id

    def __eq__(self, other) -> bool:
        """
        Check if two Message instances are equal.

        Args:
            other (Message): Another Message instance.

        Returns:
            bool: True if the two instances are equal, False otherwise.
        """
        if other is None:
            return False
        if not isinstance(other, Message):
            return False
        return (self._role, self._text) == (other._role, other.content)

    def __str__(self) -> str:
        """
        Return a YAML string representation of the message.

        Returns:
            str: YAML string representation of the message.
        """
        return yaml.dump(self, indent=4)
