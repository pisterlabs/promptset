from typing import Callable, Dict, List, Optional, Union
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from chatbot_settings import ChatBotSettings
from bot_abstract_class import BotAbstract



class BotDemoChain(BotAbstract):
    def __init__(self, chatBotSettings: ChatBotSettings()):
        self.chatbotSettings = chatBotSettings

        self.llm = chatBotSettings.llm
        self.memory = chatBotSettings.memory
        
        self.conversation_buf: ConversationChain = ConversationChain(
            llm=self.llm,
            memory=self.memory
        )

    def get_bot_response(self, text: str):
        reply = self.conversation_buf(text)
      
        return reply['response']





    
