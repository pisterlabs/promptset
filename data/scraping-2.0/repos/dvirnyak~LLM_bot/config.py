import sqlalchemy as db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType

import os
from dotenv import load_dotenv
from telegram.ext import Application

from langchain.chat_models.gigachat import GigaChat

load_dotenv()
VK_TOKEN = os.getenv('VK_TOKEN')
vk = vk_api.VkApi(token=VK_TOKEN)

PROJECT_PATH = os.path.dirname(__file__)
USERS_DB_PATH = PROJECT_PATH + "/" + os.getenv('USERS_DB_PATH')
DOCUMENTS_DB_PATH = PROJECT_PATH + "/" + os.getenv('DOCUMENTS_DB_PATH')
GIGACHAT_TOKEN = os.getenv('GIGACHAT_TOKEN')
gigachat = GigaChat(credentials=GIGACHAT_TOKEN,
                verify_ssl_certs=False)

db_engine = create_engine("sqlite:////" + USERS_DB_PATH)
db_meta = db.MetaData()
Session = sessionmaker(bind=db_engine, autoflush=False)

def prepare_dataset(documents_dir):
    for filename in os.listdir(documents_dir):
        if filename.endswith('.txt'):
            with open(documents_dir + filename, 'r') as file:
                text = file.read()

                doc = MyDoc(
                    text=text,
                    text_embedding=embeddings.embed_query(text=text)
                )


