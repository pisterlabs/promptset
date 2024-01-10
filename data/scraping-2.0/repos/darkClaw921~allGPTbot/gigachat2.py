"""Пример работы с чатом через gigachain"""
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from dotenv import load_dotenv
import os
load_dotenv()

token = os.getenv("GIGACHAT_TOKEN")
# Авторизация в сервисе GigaChat
gigaChat1 = GigaChat(credentials=token, verify_ssl_certs=False)

# messages = [
#     SystemMessage(
#         content="Ты эмпатичный бот-психолог, который помогает пользователю решить его проблемы."
#     )
# ]

# # while(True):
# # user_input = input("User: ")
# user_input = 'Какие факторы влияют на стоимость страховки на дом?'
# messages.append(HumanMessage(content=user_input))
# res = gigaChat1(messages)
# messages.append(res)
# print("Bot: ", res.content)