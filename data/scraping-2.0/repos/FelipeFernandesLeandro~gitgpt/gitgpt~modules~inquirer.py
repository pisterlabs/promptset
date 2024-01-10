import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOllama
from langchain.schema import HumanMessage


class Inquirer:
    def inquire(self, content: str):
        model = os.environ.get("MODEL")
        chat_model = ChatOllama(
            model=model,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

        messages = [HumanMessage(content=content)]
        return chat_model(messages)
