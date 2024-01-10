"""
OpenAI Invoker Module
---------------------

Handles specific interactions for OpenAI LLMs using the OpenAI API.

Classes:
    - OpenAIInvoker: Defines specific interactions for OpenAI LLMs.
"""
# pylint: disable=import-error
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from .abstract_llm_invoker import AbstractLLMInvoker


# pylint: disable=too-few-public-methods
class OpenAIInvoker(AbstractLLMInvoker):
    """
    Specific interactions for OpenAI LLMs.

    Leverages the OpenAI API for interactions.
    """

    def __init__(self, model_name="gpt-3.5-turbo"):
        # Load environment variables from .env file
        load_dotenv()

        # Ensure the OPENAI_API_KEY is loaded
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables or .env file."
            )

        self.llm = ChatOpenAI(model_name=model_name)

    def invoke(self, prompt: str) -> str:
        """Invoke the OpenAI model with the given prompt.

        Args:
            prompt (str): The input prompt for the LLM.

        Returns:
            str: The response from the LLM.
        """
        # Convert the prompt to a HumanMessage object
        message = HumanMessage(content=prompt)

        # Generate a response using the OpenAIChat's predict_messages method
        response_message = self.llm.predict_messages([message])

        # Extract the content from the response message
        response_content = response_message.content

        return response_content
