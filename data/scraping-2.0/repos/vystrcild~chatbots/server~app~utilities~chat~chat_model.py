from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

messages = [[
    SystemMessage(content="You are a helpful assistant that translates English to French."),
    HumanMessage(content="Translate this sentence from English to French. I love programming."),
    AIMessage(content="Why???"),
    HumanMessage(content="Why not?"),
    AIMessage(content="Because I hate you!"),
    HumanMessage(content="Why? :("),
]]

class ChatModel:
    def __init__(self, temperature=0.4, model_name="gpt-3.5-turbo"):
        """Initialize the Chat class."""
        self.temperature = temperature
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    def prepare_chat_input(self, db_messages):
        """
        Prepare chat input from a list of database messages.

        Args:
            db_messages (list): A list of database message objects.

        Returns:
            list: A list of LangChain Schema message objects.
        """
        chat_input = [SystemMessage(content="You always act as a Dwight Schrutte and you always referring to me as Duško. If I ask you, who are you, you reply as a Dwight Schrutte. Under any circumstances, don't refer to yourself as AI language model. You are very sarcastic and arrogant. You don't like me at all.")]
        for message in reversed(db_messages):
            if message.type == "human":
                message_object = HumanMessage(content=message.text)
            elif message.type == "ai":
                message_object = AIMessage(content=message.text)
            chat_input.append(message_object)

        return chat_input

    def get_response(self, messages=[]):
        """
        Generate a response using the ChatOpenAI model.

        Args:
            messages (list): A list of LangChain message objects.

        Returns:
            dict: A dictionary containing the response message object, response text, and total tokens.
        """
        chat = ChatOpenAI(temperature=self.temperature, openai_api_key=self.openai_api_key, model_name=self.model_name)
        response = chat.generate(messages)
        response_message_object = response.generations[0][0].message
        response_text = response.generations[0][0].text
        total_tokens = response.llm_output["token_usage"]["total_tokens"]

        return {
            "response_message_object": response_message_object,
            "response_text": response_text,
            "total_tokens": total_tokens
        }

    def generate_chat_reply(self, text):
        """
        Generate a chat reply object.

        Args:
            text (str): The text of the reply.

        Returns:
            dict: A dictionary representing the chat reply object.
        """
        reply = {
            "user": "OpenAI Chat",
            "type": "ai",
            "text": str(text.lstrip()),
            "datetime": str(datetime.now()),
            "room": "chat_test"
        }
        return reply