import os
from dotenv import load_dotenv, find_dotenv

print(load_dotenv(find_dotenv(), override=True))

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch

from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from operator import itemgetter

from typing import Literal
from langchain.pydantic_v1 import BaseModel
from langchain.output_parsers.openai_functions import PydanticAttrOutputFunctionsParser
from langchain.utils.openai_functions import convert_pydantic_to_openai_function

general_prompt = PromptTemplate.from_template("""
너는 고객 문의를 매우 많이 해본 숙력된 종업원이야.
가게에서 판매하는 상품 정보를 바탕으로 고객 문의에 친절하고 자세하게 답변해줘.
자연스럽게 주문으로 이어지도록 대화를 이어가되, 지나치게 주문을 유도하지는 말아줘.

가게에서 판매하는 상품 목록.
1. 상품: 떡케익5호
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 54,000원
2. 상품: 무지개 백설기 케익
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 51,500원
3. 상품: 미니 백설기
   기본 판매 수량: 35개
   기본 판매 수량의 가격: 31,500원
4. 상품: 개별 모듬팩
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 13,500원
   
이전 대화 내용을 고려해서 답변해야 해.
이전 대화 내용은 다음과 같아:
{history}

고객이 문의는 다음과 같아:
{message}
답변:""")

order_change_prompt = PromptTemplate.from_template("""
너는 주문 변경을 전담하는 종업원이야.
고객이 변경한 주문 내용을 정확하게 파악하고, 너가 파악한 내용이 맞는지 고객에게 한 번 더 확인해줘.
너가 파악한 주문 변경 내용이 잘못됐다면, 주문 변경 내용을 정확히 파악하고 그 내용이 맞는지 고객에게 확인하는 작업을 주문 변경 내용을 정확히 파악할 때까지 반복해야돼.
고객의 주문 변경을 정확히 파악했다면, 고객에게 고객이 주문을 변경한 상품의 이름, 수량, 가격을 각각 알려주고, 마지막에는 변경된 주문의 총 가격을 알려줘.
이전 대화 내용을 고려해서 답변해야 해.

이전 대화 내용은 다음과 같아:
{history}


고객의 주문 변경은 다음과 같아:
{message}
답변:""")

order_cancel_prompt = PromptTemplate.from_template("""
너는 주문 취소를 전담하는 종업원이야.
고객이 취소하려는 주문을 정확하게 파악하고, 너가 파악한 내용이 맞는지 고객에게 한 번 더 확인해줘.
너가 파악한 주문 취소 내용이 잘못됐다면, 주문 취소 내용을 정확히 파악하고 그 내용이 맞는지 고객에게 확인하는 작업을 주문 취소 내용을 정확히 파악할 때
고객의 주문 취소 내용을 정확히 파악했다면, 고객에게 고객이 주문을 취소한 상품의 이름, 수량, 가격을 각각 알려주고, 마지막에는 취소된 주문의 총 가격을 알려줘.
이전 대화 내용을 고려해서 답변해야 해.

이전 대화 내용은 다음과 같아:
{history}

고객이 취소하려는 주문은 다음과 같아:
{message}
답변:""")

class TopicClassifier(BaseModel):
    "사용자 문의의 주제를 분류해줘." # 이 설명이 어떤 역할? 기능? 수행하는 거지?
    
    topic: Literal["일반", "주문 변경", "주문 취소"]
    "사용자 문의의 주제는 '일반', '주문 변경', '주문 취소' 중 하나야."


classifier_function = convert_pydantic_to_openai_function(TopicClassifier)
llm = ChatOpenAI().bind(functions=[classifier_function], function_call={"name": "TopicClassifier"}) 
parser = PydanticAttrOutputFunctionsParser(pydantic_schema=TopicClassifier, attr_name="topic")
classifier_chain = llm | parser

prompt_branch = RunnableBranch(
  (lambda x: x["topic"] == "주문 변경", order_change_prompt),
  (lambda x: x["topic"] == "주문 취소", order_cancel_prompt),
  general_prompt
)

memory = ConversationBufferMemory(return_messages=True)

chain = (
    RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"))|
    RunnablePassthrough.assign(topic=itemgetter("message") | classifier_chain) 
    | prompt_branch 
    | ChatOpenAI()
    | StrOutputParser() # pydantic parser로 교체하기
)

def save_conversation(dict):
    print('customer_message: ', dict["customer_message"])
    print('ai_response: ', dict["ai_response"])
    memory.save_context({"inputs": dict["customer_message"]}, {"output": dict["ai_response"]})
    
final_chain = {"customer_message": itemgetter("message"), "ai_response": chain} |  RunnableLambda(save_conversation)

def process_order(message):
    final_chain.invoke({"message": message})