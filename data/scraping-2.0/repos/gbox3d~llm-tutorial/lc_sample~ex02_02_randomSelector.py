#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate

from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.callbacks import StreamingStdOutCallbackHandler

from random import choice
import time
import os
from dotenv import load_dotenv
load_dotenv('../.env')

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler()
    ]
)
print("chat model ready")
#%%
examples = [
    {
        "question": "프랑스에 대해 무엇을 알고 있나요?",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 파리
        언어: 프랑스어
        음식: 와인과 치즈
        화폐: 유로
        """,
    },
    {
        "question": "일본에 대해 무엇을 알고 있나요?",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 도쿄
        언어: 일본어
        음식: 초밥
        화폐: 엔
        """,    
    },
    {
        "question": "터키에 대해 무엇을 알고 있나요?",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 안카라
        언어: 터키어
        음식: 케밥
        화폐: 터키 리라
        """,    
    }
]

class RandomExampleSelector(BaseExampleSelector):
    def __init__(self, examples):
        self.examples = examples

    def add_example(self, example):
        self.examples.append(example)

    def select_examples(self, input_variables):
        return [choice(self.examples)]
    

prompt = FewShotPromptTemplate(
    example_prompt=PromptTemplate.from_template("Human: {question}\nAI:{answer}"),
    example_selector=RandomExampleSelector(examples),
    suffix="Human: {country} 에 대해 무엇을 알고 있나요?\nAI:",
    input_variables=["country"],
)

_p = prompt.format(
    country="한국"
)

print(_p)
#%%

chain = prompt | chat

answer = chain.invoke({"country": "한국"})

print(f"\n\nanswer : {answer.content}")
    