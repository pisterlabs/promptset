#%%
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage,AIMessage,SystemMessage

import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')} ")
print(f"OpenAI Model Name: {os.getenv('OPENAI_MODEL_NAME')} ")

chat = ChatOpenAI(temperature=0.1)

print('llm ready')
#%%

start_tick = time.time()
language = "korean"
name = "doromiss"
country_a = "korea"
country_b = "japan"

messages = [
    SystemMessage(
        content=f"You are a geography expert. And you only reply in {language}.",
    ),
    AIMessage(content=f"Ciao, mi chiamo {name}!"),
    HumanMessage(
        content=f"What is the distance between {country_a} and {country_b}. Also, what is your name?",
    ),
]



answer = chat.predict_messages(messages)
print(f'elapsed time: {time.time() - start_tick}')

print( type(answer))
print(answer)
print(answer.content)

# %%
