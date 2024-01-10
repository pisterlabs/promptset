#%%
from langchain.chat_models import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate

from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

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

#%%
example_prompt = PromptTemplate.from_template("Human: {question}\nAI:{answer}")

prompt = FewShotPromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
    suffix="Human: {country} 에 대해 무엇을 알고 있나요?\nAI:",
    input_variables=["country"],
)

chain = prompt | chat

#%%
_pstr = prompt.format(
    country="한국"
)  
print(_pstr)
#%%
chain.invoke({"country": "한국"})

#%%
examples = [
    {
        "country": "프랑스",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 파리
        언어: 프랑스어
        음식: 와인과 치즈
        화폐: 유로
        """,
    },
    {
        "country": "일본",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 도쿄
        언어: 일본어
        음식: 초밥
        화폐: 엔
        """,    
    },
    {
        "country": "터키",
        "answer": """
        제가 알고 있는 것은 다음과 같습니다:
        수도: 안카라
        언어: 터키어
        음식: 케밥
        화폐: 터키 리라
        """,    
    }
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", " {country} 에 대해 무엇을 알고 있나요?"),
        ("ai", "{answer}"),
    ]
)

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 지리 전문가입니다. 그리고 {language}로만 답변합니다. 가능한 짤막한 답변을 해주세요."),
        example_prompt,
        ("human", "What do you know about {country}?"),
    ]
)


chain = final_prompt | chat

chain.invoke({"country": "중국", "language": "한국어"})

# %%

_pstr = final_prompt.format(
    country="중국",
    language="한국어",
)

print(_pstr)


# %%
