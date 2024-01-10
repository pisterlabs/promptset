from typing import Callable, Dict, List, Optional, Union
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from chatbot_settings import ChatBotSettings
from bot_abstract_class import BotAbstract



class BotConversationChain(BotAbstract):
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
      
        questions = [
             {
                "question": "What is your name?",
                "type": "open-ended"
            },
            {
                "question": "What hurts?",
                "type": "open-ended"
            },
            {
                "question": "What is your level of pain?",
                "type": "numerical"
            },
            {
                "question": "Do you have any other symptoms?",
                "type": "open-ended"
            }
        ]


        return reply['response']





    
