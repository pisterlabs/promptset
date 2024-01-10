#%%
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI

import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

print(f"OpenAI API Key: {os.getenv('OPENAI_API_KEY')} ")
print(f"OpenAI Model Name: {os.getenv('OPENAI_MODEL_NAME')} ")

llm = OpenAI()
chat = ChatOpenAI()

print('llm ready')

#%% 배이스 모델로부터 답변을 생성하는 예제
start_tick = time.time()
a = llm.predict("How many planets are there in the solar system?")
print(f'elapsed time: {time.time() - start_tick}')

print(f'answer type: {type(a)}')
print(a)

#%% 쳇봇 모델로부터 답변을 생성하는 예제
start_tick = time.time()
b = chat.predict("How many planets are there in the solar system?")
print(f'elapsed time: {time.time() - start_tick}')
print(b)
# %%
