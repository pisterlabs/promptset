import threading

from langchain import ConversationChain
from langchain.callbacks import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from callbacks.streaming import ThreadedGenerator, ChainStreamHandler
import os
from dotenv import load_dotenv

class SimpleConversationChat:
    def __init__(self, history):
        load_dotenv()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.set_memory(history)

    def set_memory(self, history):
        for message in history:
            if message.role == 'assistant':
                self.memory.chat_memory.add_ai_message(message.content)
            else:
                self.memory.chat_memory.add_user_message(message.content)

    def generator(self, user_message):
        g = ThreadedGenerator()
        threading.Thread(target=self.llm_thread, args=(g, user_message)).start()
        return g

    def llm_thread(self, g, user_message):
        try:
            llm = ChatOpenAI(
                verbose=True,
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                temperature=0.7,
            )
            conv = ConversationChain(
                llm=llm,
                memory=self.memory,
            )
            conv.predict(input=user_message)
        finally:
            g.close()
