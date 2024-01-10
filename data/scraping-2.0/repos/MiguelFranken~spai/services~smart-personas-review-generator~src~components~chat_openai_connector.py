from langchain.chat_models import ChatOpenAI


class ChatOpenAIConnector:
    """
    A connector class for OpenAI's GPT-3.5-turbo language model.

    This class provides an interface for interacting with the GPT-3.5-turbo language model through the OpenAI API.
    The `llm` attribute is an instance of the `ChatOpenAI` class, which is used to generate text based on prompts.

    Attributes:
        llm (ChatOpenAI): An instance of the ChatOpenAI class representing the GPT-3.5-turbo language model.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
        )

    @property
    def get_llm(self):
        """
        Returns the ChatOpenAI language model instance associated with this connector.

        Returns:
            ChatOpenAI: An instance of the ChatOpenAI class representing the GPT-3.5-turbo language model.
        """
        return self.llm
