from dotenv import dotenv_values
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from src.chat.boilerplate_ai import AI
from src.chat.system_message import SYSTEM_MESSAGE
from src.db.db import DBMessage

config = dict(dotenv_values(".env"))

model = config["MODEL"]
api_key = config["OPENAI_API_KEY"]


class OpenAI(AI):
    def __init__(self) -> None:
        self.SYSTEM_MESSAGE = SystemMessage(content=SYSTEM_MESSAGE)

    def create_messages(self, conversation: list[DBMessage] = []):
        return [
            AIMessage(content=message.content)
            if message.type == "ai"
            else HumanMessage(content=message.content)
            for message in conversation
        ]

    def respond(self, message_content: str, conversation: list[DBMessage] = []) -> str:
        chat = ChatOpenAI(model_name=model, openai_api_key=api_key)

        conversation_messages = self.create_messages(conversation)

        all_messages = (
            [self.SYSTEM_MESSAGE]
            + conversation_messages
            + [HumanMessage(content=message_content)]
        )

        message = chat(all_messages)

        return message.content
