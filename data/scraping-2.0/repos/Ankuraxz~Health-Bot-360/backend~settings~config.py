import logging
import os

import pymongo
from langchain.llms import OpenAIChat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    _instance = None

    @staticmethod
    def get_instance():
        """Return the singleton instance of Config"""
        if Config._instance is None:
            Config._instance = Config()
        return Config._instance

    def __init__(self):
        if Config._instance is not None:
            raise Exception(
                "Config is a singleton class, use get_instance() to get the instance."
            )

        try:
            self.open_ai_key = os.environ.get('OPENAI_API_KEY')
            self.mongodb_uri = os.environ.get('MONGO_URI')
        except Exception as e:
            logger.error(f"An Exception Occurred while loading config --> {e}")
            raise Exception(f"An Exception Occurred while loading config --> {e}")

    def get_openai_chat_connection(self):
        try:
            logger.info(f"Connecting to Open AI 3.5 chat")
            chat_llm = OpenAIChat(max_tokens=4000, temperature=0.3, model='gpt-3.5-turbo-16k')
            return chat_llm
        except Exception as e:
            logger.error(f"An Exception Occurred while connecting to Open AI --> {e}")
            raise Exception(f"An Exception Occurred while connecting to Open AI --> {e}")

    def get_mongodb_connection(self):
        try:
            logger.info(f"Connecting to MongoDB")
            client = pymongo.MongoClient(self.mongodb_uri)
            try:
                client.admin.command('ping')
                logger.info(f"Connected to MongoDB")
                return self.mongodb_uri
            except Exception as e:
                logger.error(f"An Exception Occurred while connecting to MongoDB --> {e}")
                raise Exception(f"An Exception Occurred while connecting to MongoDB --> {e}")

        except Exception as e:
            logger.error(f"An Exception Occurred while connecting to MongoDB --> {e}")
            raise Exception(f"An Exception Occurred while connecting to MongoDB --> {e}")
