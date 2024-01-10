#%%
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,PromptTemplate
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

template = PromptTemplate.from_template(
    """
    {country_a}와 {country_b}의 거리는 얼마나 되나요?
    한국어로만 답변해주세요.
    """
)
prompt = template.format(country_a="한국", country_b="일본")
answer = chat.predict(prompt)
print(f'elapsed time: {time.time() - start_tick}')
print( type(answer))
print(answer)
# %%
template = ChatPromptTemplate.from_messages( [
    ("system" , "당신은 지리 전문가입니다. 그리고 {language}로만 답변합니다."),
    ("ai" ,"안녕하세요! 저는 {name}입니다."),
    ("human" ,"{country_a}와 {country_b}의 거리는 얼마나 되나요? 그리고 당신의 이름은 무엇인가요?"),
])

prompt = template.format_messages(
    language="한국어",
    name="도로미스",
    country_a="한국",
    country_b="일본",
)

print(prompt)
for _prompt in prompt : 
    print(_prompt)


# %%
start_tick = time.time()
answer = chat.predict_messages(prompt)
print(f'elapsed time: {time.time() - start_tick}')
print( type(answer))
print(answer)


# %%
