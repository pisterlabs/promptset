from langchain.schema import BaseMessage


class LMIMessage(BaseMessage):
    """A Message from the LMI."""

    example: bool = False
    """Whether this Message is being passed in to the model as part of an example 
        conversation.
    """

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "lmi"
