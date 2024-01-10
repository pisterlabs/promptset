#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser

import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

chat = ChatOpenAI(temperature=0.8)

#%%
class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))
    
#%%
p = CommaOutputParser()
p.parse("a, b, c")
# %%

# 한글로 콤마로 구분된 리스트를 만들라고 하면 가끔 명령을 무시한다. 영문은 잘 작동한다.
templete = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a list generating machine. \n
    Everything you are asked will be answered with a comma separated list of max {max_items} in korean \n
    Do NOT reply with anything else.
            """
        ),
        ("human", "{question}")
    ])

#%%
start_tick = time.time()
prompt = templete.format_messages(
    max_items=10,
    question="한류 걸구룹 이름을 몇개 만들어줘")

answer = chat.predict_messages(prompt)

print(f'elapsed time: {time.time() - start_tick}')
print(answer)
p.parse(answer.content)

# %%
start_tick = time.time()
chain = templete | chat | CommaOutputParser()

answer = chain.invoke({
    "max_items": 5,
    "question": "한류 걸그룹 이름을 지어줘"
})

print(f'elapsed time: {time.time() - start_tick}')
print(answer)
# %%
