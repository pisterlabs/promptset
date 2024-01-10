import os
from typing import Dict, Type
from langchain.chat_models import ChatOpenAI
import openai
from bot_abstract_class import BotAbstract


class ChatBotSettings:
    
    def __init__(self, memory = None, llm= None, tools = None):
        self.memory = memory
        self.llm = llm
        self.tools = tools
        self.set_environment_variables()
        self.users = self.get_users()

    def set_environment_variables(self):
        os.environ["PINECONE_API_KEY"] = ''
        os.environ["PINECONE_API_ENV"] = ''
        os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
        os.environ["WOLFRAM_ALPHA_APPID"] = ""
        os.environ["SERPAPI_API_KEY"] = ""
        os.environ["OPENAI_ORGANIZATION_ID"] = ""
        os.environ["DISCORD_BOT_TOKEN"] = ""
        os.environ["KNOWLEDGE_BASE_FILE"] = ""
        os.environ["COHERE_API_KEY"] = "YOUR_API_KEY"
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "YOUR_API_KEY"
    
    @classmethod
    def PINECONE_API_KEY(cls):
        return os.environ.get("PINECONE_API_KEY")

    @classmethod
    def PINECONE_API_ENV(cls):
        return os.environ.get("PINECONE_API_ENV")

    @classmethod
    def OPENAI_API_KEY(cls):
        return os.environ.get("OPENAI_API_KEY")

    @classmethod
    def WOLFRAM_ALPHA_APPID(cls):
        return os.environ.get("WOLFRAM_ALPHA_APPID")

    @classmethod
    def SERPAPI_API_KEY(cls):
        return os.environ.get("SERPAPI_API_KEY")

    @classmethod
    def OPENAI_ORGANIZATION_ID(cls):
        return os.environ.get("OPENAI_ORGANIZATION_ID")

    @classmethod
    def DISCORD_BOT_TOKEN(cls):
        return os.environ.get("DISCORD_BOT_TOKEN")
    
    @classmethod
    def COHERE_API_KEY(cls):
        return os.environ.get("COHERE_API_KEY")
    
    @classmethod
    def HUGGING_FACE_API_KEY(cls):
        return os.environ.get("HUGGINGFACEHUB_API_TOKEN")
       
    @classmethod
    def KNOWLEDGE_BASE_FILE(cls):
        return "knowledge_base.json"

    def get_openai_models():
        model_list = openai.Model.list()['data']
        model_ids = [x['id'] for x in model_list]
        model_ids.sort()
        pprint.pprint(model_ids)

    def get_users(self):
        users = {
            "chatbot": "vernalfuture",
            "user2": "anotheruser"
        }
        return users

