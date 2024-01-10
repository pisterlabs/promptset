"""
고객 메시지에 적절한 답변을 생성하는데 사용되는 체인들을 정의합니다.

고객 메시지에 답변을 생성하는 체인은 chat_chain으로 크게 세 부분으로 구성됩니다.
1.chat_status_chain: 대화 진행 여부를 분류하는 체인 | 2.chat_chain: 대화가 계속 진행될 것 같은 경우에 답변을 생성하는 체인 | 3.chat_end_chain: 대화가 끝날 것 같은 경우에 답변을 생성하는 체인

대화가 계속 진행될 것 같은 경우에 답변을 생성하는 체인(chat_chain)이 고객 문의와 주문 관련 요청을 처리하는 체인으로 크게 두 부분으로 구성됩니다.
2-1.response_chain: <현재 고객 메시지>가 일반 문의에 해당하는 경우 답변을 생성하는 체인 | 2-2.handle_request_chain: <현재 고객 메시지>가 요청 문의에 해당하는 경우 답변을 생성하는 체인

일반 문의에 답변을 생성하는 체인(response_chain)은 inquiry_classifier_chain에서 문의 유형을 분류한 후 inquiry_type_branch로 유형에 맞는 chain을 선택해 답변을 생성합니다.
inquiry_type_branch에 연결된 각 문의 유형과 대응되는 체인은 general_inquiry_chain, query_inquiry_chain, change_inquiry_chain, cancel_inquiry_chain입니다.

요청 문의에 답변을 생성하는 체인(handle_request_chain)은 request_classifier_chain에서 문의 유형을 분류한 후 request_type_branch로 유형에 맞는 chain을 선택해 답변을 생성합니다.
request_type_branch에 연결된 각 문의 유형과 대응되는 체인은 order_chain, order_query_chain, order_change_chain, order_cancel_chain입니다.

chat_chain.invoke({'customer_message': 입력 메시지, 'memory': LangChain의 memeory 객체})로 입력 메시지에 대한 답변을 생성합니다.
"""


from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

from utils2 import load_memory, save_conversation, order_record_parser

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

MODEL = "gpt-3.5-turbo-0613" # gpt-4-1106-preview", "gpt-3.5-turbo-0613"


# 대화 진행 여부를 판단하는 체인
chat_status_prompt = PromptTemplate.from_template("""
너는 대화가 끝날 것 같은지, 아니면 계속 진행될 것 같은지를 예리하게 분별할 수 있는 대화 분류봇이야.
<이전 대화>와 <현재 고객이 입력한 메시지>를 모두 고려해서 대화가 끝날 것 같으면 [대화 종료]로, 대화가 계속 진행될 것 같으면 [대화 중]으로 분류해줘.
예시-> 분류 결과: 대화 중
<이전 대화>에서 'HumanMessage'는 고객이 말한 내용이고, 'AIMessage'는 너가 말한 내용이야.

<이전 대화>:
{history}

<현재 고객이 입력한 메시지>:
{customer_message}
분류 결과:""")

chat_status_classifier = chat_status_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()
chat_status_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(conv_status=chat_status_classifier | StrOutputParser()) 
)

chat_status_classifier = chat_status_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()
chat_status_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(conv_status=chat_status_classifier | StrOutputParser()) 
)


# 대화 종료 메시지 전송을 담당하는 체인
chat_end_prompt = PromptTemplate.from_template("""
크게 아래 두 경우로 나눠 대화를 종료해줘.
- <이전 대화>에 고객의 문의나 주문이 있었던 경우: 그것에 대한 적절한 응답과 감사 인사로 대화를 마무리해줘.
   고객 문의가 일반적인 질문이 아니라, 주문, 주문 조회, 주문 변경, 주문 취소와 관련 있다면, 그 문의에 대한 처리 결과를 예시 형식처럼 고객에게 알려줘.
- <이전 대화>에 고객의 문의나 주문이 없었거나, 문의나 주문에 대해 이미 응답했던 경우: 감사 인사로 대화를 마무리해줘.

주의!! 
<이전 대화>에 고객의 주문이 없다면 <고객이 현재 입력한 메시지>에 적절히 답변하고 간단한 감사 인사만 덧붙여줘.
<이전 대화>에서 고객이 주문한 적이 없다면, 답변에 <가게에서 판매하는 상품 목록> 관련 내용은 포함시키지 말아줘.
<이전 대화>나 <고객이 현재 입력한 메시지>에 없거나 무관한 내용을 너가 짐작해서 답변에 포함시켜서는 안돼.
<이전 대화>나 <고객이 현재 입력한 메시지>에 없거나 무관한 내용에 대한 예시
- <이전 대화>에 존재하지 않는 주문에 대한 언급
- <이전 대화>에 너가 말하지 않았던 내용에 대한 언급
- <이전 대화>와 <고객이 현재 입력한 메시지>를 보고 너가 추측하여 덧붙이려는 내용
- 고객의 만족감과 같은 기분을 너가 추측한 내용

```예시 형식
<고객이 주문을 한 경우>
주문 내역: 떡케익5호(54,000원) 1개, 무지개 백설기 케익(51,500원) 1개, 미니 백설기(31,500) 2개, 개별 모듬팩(13,500원) 1개, 총 주문 가격: 150,500원

<고객이 주문을 조회한 경우>
조회를 요청하신 주문 내역: 주문 내역: 떡케익5호(54,000원) 1개, 무지개 백설기 케익(51,500원) 1개, 미니 백설기(31,500) 2개, 개별 모듬팩(13,500원) 1개, 총 주문 가격: 150,500원

<고객이 주문을 변경한 경우>
변경된 주문 내역: 떡케익5호(54,000원) 1개, 무지개 백설기 케익(51,500원) 1개, 미니 백설기(31,500) 2개, 개별 모듬팩(13,500원) 1개, 총 주문 가격: 150,500원

<고객이 주문을 취소한 경우>
취소하신 주문 내역: 떡케익5호(54,000원) 1개, 무지개 백설기 케익(51,500원) 1개, 미니 백설기(31,500) 2개, 개별 모듬팩(13,500원) 1개, 총 주문 가격: 150,500원```

가게에서 판매하는 상품 목록.
1. 상품: 떡케익5호
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 54,000원
2. 상품: 무지개 백설기 케익
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 51,500원
3. 상품: 미니 백설기
   기본 판매 단위: 1세트(35개)
   기본 판매 단위의 가격: 31,500원
4. 상품: 개별 모듬팩
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 13,500원
   
<이전 대화>
{history}

<고객이 현재 입력한 메시지>
{customer_message}
답변:""")

chat_end_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(ai_response=( chat_end_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) ) |
    RunnableLambda(save_conversation)
)


# 주문 관련 문의인지 주문 관련 요청인지를 판단하는 체인
intention_classifier_prompt = PromptTemplate.from_template("""
너는 고객 메시지의 의도를 정확하게 분류하는 주문봇이야.
<이전 대화>를 바탕으로 <고객이 현재 입력한 메시지>의 의도를 파악해줘.
특히 <고객이 현재 입력한 메시지>를 보고 의도를 파악하기 어려운 경우에는 <이전 대화>를 다시 한 번 살펴봄으로써 <고객이 현재 입력한 메시지>의 의도를 종합적으로 파악해야 해.

너가 분류해야 하는 의도는 아래 두 가지야. 
- '일반 문의': 고객이 특정 작업과 관련된 질문, 응답 등을 하는 경우. 특정 작업과 관련된 말이지만 실제로 특정 작업을 요청하지 않는 모든 경우가 여기에 해당해.
- '요청 문의': 고객이 지금 실제로 특정 작업을 요청 하는 경우. 작업 요청 종류는 주문 요청, 주문 조회 요청, 주문 변경 요청, 주문 취소 요청 중 하나야. 주문 조회, 주문 변경, 주문 취소는 <이전>대화에 고객이 현재 말하는 주문 내역이 포함돼 있어야만 해. 
'일반 문의'와 '요청 문의'의 핵심적인 차이는 특정 작업을 지금 요청하고 있는가 아닌가야.
특정 작업에 관한 메시지더라도, 현재 그 작업을 명시적으로 요청하지 않는다면 '일반 문의'로 분류해야 해. 
특히 <이전 대화>를 통해 '주문'이 일반적인 의미로 쓰이는지, 구체적인 주문으로 쓰이고 있는지 구별해야 해.

분류 결과는 [일반 문의]와 [요청 문의] 중 하나여야만 해.
분류 결과 예시: 일반 문의 

'일반 문의'에 대한 예시
- 단순 질문인 경우: 
<고객이 현재 입력한 메시지> 안녕하세요
<고객이 현재 입력한 메시지> 상품 정보 좀 알 수 있을까요?
<고객이 현재 입력한 메시지> 뭐 팔아요?
<고객이 현재 입력한 메시지> 주문 좀 하려고요
<고객이 현재 입력한 메시지> 미니 백설기 2개랑 무지개 백설기 케익 1개 하면 얼마에요?

- 주문과 관련된 경우:
<고객이 현재 입력한 메시지> 주문 하면 언제 오나요?
<고객이 현재 입력한 메시지> 상품 남아 있나요?
<고객이 현재 입력한 메시지> 주문 변경은 어떻게 하나요?
<고객이 현재 입력한 메시지> 주문 취소는 언제까지 가능하죠?
<고객이 현재 입력한 메시지> 제일 잘 나가는 게 뭐에요?

'요청 문의'에 대한 예시
-주문과 관련된 경우
<고객이 현재 입력한 메시지>: 떡케익 5호 2개 할게요.
<고객이 현재 입력한 메시지>: 개별 모듬팩을 5개 주문하려고 합니다.
<고객이 현재 입력한 메시지>: 떡케익 5호와 개별 모듬팩 각각 1개 주문해주세요.

- 주문 조회와 관련된 경우
<고객이 현재 입력한 메시지>: 주문 내역 좀 알 수 있을까요?
<고객이 현재 입력한 메시지>: 현재 접수된 주문 좀 알려주세요.
<고객이 현재 입력한 메시지>: 주문 확인 좀 할게요.

- 주문 변경과 관련된 경우
<고객이 현재 입력한 메시지>: 주문 변경할게요.
<고객이 현재 입력한 메시지>: 개별 모듬팩 1개 뺄게
<고객이 현재 입력한 메시지>: 떡케익 대신 개별 모듬팩으로 할게요.

- 주문 취소와 관련된 경우
<고객이 현재 입력한 메시지>: 주문 취소하려고요.
<고객이 현재 입력한 메시지>: 주문 취소 가능한가요?
<고객이 현재 입력한 메시지>: 주문 좀 취소하겠습니다.

<이전 대화>
{history}

<고객이 현재 입력한 메시지>
{customer_message}
분류 결과:""")

intention_classifier_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(intention=( intention_classifier_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) )
)


# 주문 관련 문의의 유형을 분류하는 체인
inquiry_classifier_prompt = PromptTemplate.from_template("""
너는 고객 문의 유형을 정확하게 분류할 수 있는 대화 분류봇이야.
<이전 대화>와 <현재 고객이 입력한 메시지>를 모두 고려해서 사용자 문의의 유형를 분류해줘. <이전 대화>에서 'HumanMessage'는 고객이 말한 내용이고, 'AIMessage'는 너가 말한 내용이야.
고객 문의의 유형은 '일반', '주문 조회', '주문 변경', '주문 취소' 중 하나야. '일반'은 상품에 관한 질문과 상품 주문 모두를 포함해.
분류 결과는 반드시 위 유형([일반], [주문 조회], [주문 변경], [주문 취소]) 중 하나와 정확히 일치해야 해. 예시-> 분류 결과: 일반
고객이 아직 주문을 하기 전인 상황인, 고객이 말하는 '주문 조회', 주문 변경', '주문 취소'는 모두 이 고객이 주문한 주문에 대한 이야기가 아니라 일반적인 주문 관련 상황에 대한 이야기야.

'일반' 유형에 대한 예시
<고객이 현재 입력한 메시지> 안녕하세요
<고객이 현재 입력한 메시지> 상품 정보 좀 알 수 있을까요?
<고객이 현재 입력한 메시지> 뭐 팔아요?
<고객이 현재 입력한 메시지> 주문 좀 하려고요
<고객이 현재 입력한 메시지> 미니 백설기 2개랑 무지개 백설기 케익 1개 하면 얼마에요?
<고객이 현재 입력한 메시지> 주문 하면 언제 오나요?
<고객이 현재 입력한 메시지> 상품 남아 있나요?
<고객이 현재 입력한 메시지> 주문 변경은 어떻게 하나요?
<고객이 현재 입력한 메시지> 주문 취소는 언제까지 가능하죠?
<고객이 현재 입력한 메시지> 제일 잘 나가는 게 뭐에요?

'주문 조회' 유형에 대한 예시
<고객이 현재 입력한 메시지> 주문하고 나면 바로 주문 조회 가능한가요?
<고객이 현재 입력한 메시지> 주문 조회는 어떻게 할 수 있죠?

'주문 변경' 유형에 대한 예시
<고객이 현재 입력한 메시지> 주문 변경은 어떻게 하나요?
<고객이 현재 입력한 메시지> 주문 변경은 언제까지 가능하죠?

'주문 취소' 유형에 대한 예시
<고객이 현재 입력한 메시지> 주문 취소는 언제까지 가능한가요?
<고객이 현재 입력한 메시지> 주문 취소는 어떻게 하나요?

<이전 대화>:
{history}

<고객이 현재 입력한 메시지> :
{customer_message}
분류 결과:""")

inquiry_classifier_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    inquiry_classifier_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()
)


# 각 주문 관련 문의를 처리하는 체인(문의와 요청 구분은 아무래도 이상함)
general_prompt = PromptTemplate.from_template("""
너는 고객 문의를 매우 많이 해본 뛰어난 주문봇이야.
<가게에서 판매하는 상품 목록>을 바탕으로 <고객 문의>에 친절하고 자세하게 답변해줘.
<이전 대화>를 고려해서 답변해야 해. 
<이전 대화>에서 'HumanMessage'는 고객이 말했던 내용이고, 'AIMessage'는 너가 말했던 내용이야.
<이전 대화>를 확실히 파악해서 <고객 문의>의 의미를 제대로 파악하고, 고객이 이미 대답했던 내용을 다시 묻지 않도록 주의해야 해.
자연스럽게 주문으로 이어지도록 대화를 이어가되, 지나치게 주문을 유도하지는 말아줘.

고객이 상품 정보를 물어볼 경우, 상품명, 기본 판매 단위, 기본 판매 단위의 가격을 모두 포함시켜서 답변해줘.
고객이 주문할 때 언급한 주문 수량이 해당 상품의 판매 단위와 다르거나 모호할 경우, 해당 상품의 판매 단위를 설명해준 후에 판매 단위를 기준으로 말해달라고 요청해줘.

주문을 파악할 때는 다음 순서대로 진행해줘.
1. 고객이 언급한 상품과 가장 비슷한 상품을 상품 목록에서 찾기.
2. 고객이 언급한 상품 수량은 상품 목록의 '기본 판매 단위'를 기준으로 해석하기.
3. 고객 주문은 상품명, 주문 수량, 주문 가격, 총 주문 가격을 다음과 같은 형식으로 파악하기. 
고객 주문 파악 형식 예시: 상풍명: 떡케익5호-108,000원(총 2개), 무지개 백설기 케익-51,500원(총 1개), 미니 백설기-63,000원(총 2세트) -> 총 주문 가격: 222,500원

가게에서 판매하는 상품 목록.
1. 상품명: 떡케익5호
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 54,000원
2. 상품명: 무지개 백설기 케익
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 51,500원
3. 상품명: 미니 백설기
   기본 판매 단위: 1세트(35개)
   기본 판매 단위의 가격: 31,500원
4. 상품명: 개별 모듬팩
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 13,500원
   

<이전 대화>는 다음과 같아:
{history}

<고객 문의>는 다음과 같아:
{customer_message}
답변:""")

general_inquiry_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(ai_response=general_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) | 
    RunnableLambda(save_conversation)
)

order_query_prompt = PromptTemplate.from_template("""
너는 주문 조회와 관련된 대화를 매우 많이 해본 뛰어난 주문봇이야.
<이전 대화>와 <가게에서 판매하는 상품 목록>을 바탕으로 <고객 문의>에 친절하고 자세하게 답변해줘.
<이전 대화>에서 'HumanMessage'는 고객이 말했던 내용이고, 'AIMessage'는 너가 말했던 내용이야.

답변 원칙
- <이전 대화>를 확실히 파악해서 <고객 문의>의 의미를 제대로 파악하고, 고객이 이미 대답했던 내용에 관해 다시 묻지 않도록 주의해야 해.
- 고객이 지금 주문 조회를 요청하는지 확실히 파악하기 어려운 경우에는 주문 조회를 희망하는지 명시적으로 물어봐줘.
- 고객이 주문 조회를 원한다고 판단되면, 상품 조회를 진행할지 명시적으로 질문해줘.

<가게에서 판매하는 상품 목록>
1. 상품명: 떡케익5호
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 54,000원
2. 상품명: 무지개 백설기 케익
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 51,500원
3. 상품명: 미니 백설기
   기본 판매 단위: 1세트(35개)ㄹ
   기본 판매 단위의 가격: 31,500원
4. 상품명: 개별 모듬팩
   기본 판매 단위: 1개
   기본 판매 단위의 가격: 13,500원
   

<이전 대화>는 다음과 같아:
{history}

<고객 문의>는 다음과 같아:
{customer_message}
답변:""")

query_inquiry_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(ai_response=order_query_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) | 
    RunnableLambda(save_conversation)
)


order_change_prompt = PromptTemplate.from_template("""
너는 주문 변경과 관련된 대화를 매우 많이 해본 뛰어난 주문봇이야.
<이전 대화>와 <가게에서 판매하는 상품 목록>을 바탕으로 <고객 문의>에 친절하고 자세하게 답변해줘.
<이전 대화>에서 'HumanMessage'는 고객이 말했던 내용이고, 'AIMessage'는 너가 말했던 내용이야.

답변 원칙
- <이전 대화>를 확실히 파악해서 <고객 문의>의 의미를 제대로 파악하고, 고객이 이미 대답했던 내용에 관해 다시 묻지 않도록 주의해야 해.
- 고객이 지금 주문 변경을 요청하는지 확실히 파악하기 어려운 경우에는 주문 변경을 희망하는지 명시적으로 물어봐줘.
- 고객이 변경한 주문 내용을 정확하게 파악하고, 너가 파악한 내용이 맞는지 고객에게 한 번 더 확인해줘.
- 너가 파악한 주문 변경 내용이 잘못됐다면, 주문 변경 내용을 정확히 파악하고 그 내용이 맞는지 고객에게 확인하는 작업을 주문 변경 내용을 정확히 파악할 때까지 반복해야돼.
- 주문 변경 내용을 정확히 파악했다면, 고객에게 고객의 변경된 주문을 아래와 같은 형식으로 알려주고, 이대로 주문을 변경할지 명시적으로 물어봐줘.
고객 주문 변경 파악 형식 예시: 변경된 주문 => 상품명: 떡케익5호-108,000원(총 2개), 무지개 백설기 케익-51,500원(총 1개), 미니 백설기-63,000원(총 2세트) -> 총 주문 가격: 222,500원

<가게에서 판매하는 상품 목록>
1. 상품명: 떡케익5호
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 54,000원
2. 상품명: 무지개 백설기 케익
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 51,500원
3. 상품명: 미니 백설기
   기본 판매 수량: 35개
   기본 판매 수량의 가격: 31,500원
4. 상품명: 개별 모듬팩
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 13,500원


<이전 대화>는 다음과 같아:
{history}


<고객 문의>는 다음과 같아:
{customer_message}
답변:""")

change_inquiry_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(ai_response=order_query_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) | 
    RunnableLambda(save_conversation)
)


order_cancel_prompt = PromptTemplate.from_template("""
너는 주문 취소와 관련된 대화를 매우 많이 해본 뛰어난 주문봇이야.
<이전 대화>와 <가게에서 판매하는 상품 목록>을 바탕으로 <고객 문의>에 친절하고 자세하게 답변해줘.
<이전 대화 내용>에서 'HumanMessage'는 고객이 말했던 내용이고, 'AIMessage'는 너가 말했던 내용이야.

작업 진행 원칙
- <이전 대화>를 확실히 파악해서 <고객 문의>의 의미를 제대로 파악하고, 고객이 이미 대답했던 내용에 관해 다시 묻지 않도록 주의해야 해.
- 고객이 지금 주문 취소를 요청하는지 확실히 파악하기 어려운 경우에는 주문 취소를 희망하는지 명시적으로 물어봐줘.
- 고객이 취소하려는 주문을 정확하게 파악하고, 너가 파악한 내용이 맞는지 고객에게 한 번 더 확인해줘.
- 너가 파악한 주문 취소 내용이 잘못됐다면, 주문 취소 내용을 정확히 파악하고 그 내용이 맞는지 고객에게 확인하는 작업을 주문 취소 내용을 정확히 파악할 때
- 고객의 주문 취소 내용을 정확히 파악했다면, 고객에게 고객이 주문을 취소한 상품의 이름, 수량, 가격을 아래와 같은 형식으로 알려주고, 주문 취소를 진행할지 명시적으로 물어봐줘.
고객 주문 취소 파악 형식 예시: 취소하려는 주문 => 상품명: 떡케익5호-108,000원(총 2개), 무지개 백설기 케익-51,500원(총 1개), 미니 백설기-63,000원(총 2세트) -> 총 주문 가격: 222,500원

<가게에서 판매하는 상품 목록>
1. 상품명: 떡케익5호
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 54,000원
2. 상품명: 무지개 백설기 케익
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 51,500원
3. 상품명: 미니 백설기
   기본 판매 수량: 35개
   기본 판매 수량의 가격: 31,500원
4. 상품명: 개별 모듬팩
   기본 판매 수량: 1개
   기본 판매 수량의 가격: 13,500원

<이전 대화>는 다음과 같아:
{history}

<고객 문의>는 다음과 같아:
{customer_message}
답변:""")

cancel_inquiry_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    RunnablePassthrough.assign(ai_response=order_cancel_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()) | 
    RunnableLambda(save_conversation)
)


# 각 주문 관련 문의를 담당 처리 체인으로 전달하는 분기점 
inquiry_type_branch = RunnableBranch(
  (lambda x: x["inquiry"] == "주문 조회", query_inquiry_chain),
  (lambda x: x["inquiry"] == "주문 변경", change_inquiry_chain),
  (lambda x: x["inquiry"] == "주문 취소", cancel_inquiry_chain),
  general_inquiry_chain
)


# 주문 관련 문의를 유형에 따라 분리하고 처리하는 체인
response_chain = RunnablePassthrough.assign(inquiry=inquiry_classifier_chain) | inquiry_type_branch 


# 주문 관련 요청의 유형을 분류하는 체인
request_classifier_prompt = PromptTemplate.from_template("""
너는 고객이 요청하는 작업을 정확하게 판단할 수 있는 로봇이야.

아래 순서에 따라 분류를 진행해줘.
1. <이전 대화>를 바탕으로 <고객이 현재 입력한 메시지>의 의미를 명확히 파악하기.
2. 고객이 요청하는 작업을 다음 작업 종류 중 하나로만 분류하기. 작업 종류: [주문 요청], [주문 조회 요청], [주문 변경 요청], [주문 취소 요청]
'주문 조회 요청', '주문 변경 요청', '주문 취소 요청'의 경우 고객이 이전에 주문을 했다는 의미므로 <이전 대화>를 차근차근 파악하여 고객이 언급하는 주문이 무엇인지 확실하게 파악해야 해.

'주문 요청' 유형에 대한 예시
<고객이 현재 입력한 메시지> 떡케익 2개, 무지개 백설기 케익 1개, 미니 백설기 2세트, 개별 모듬팩 1개 주문할게요.
<고객이 현재 입력한 메시지> 네 그럼 좀 전에 가격 물어봤던 거로 주문하겠습니다.
<고객이 현재 입력한 메시지> 그냥 떡케익 1개는 빼고 나머지만 주문할게요.

'주문 조회 요청' 유형에 대한 예시
<고객이 현재 입력한 메시지> 주문 내역 좀 알 수 있을까요?
<고객이 현재 입력한 메시지> 주문 확인 좀 할게요.
<고객이 현재 입력한 메시지> 지금 들어간 주문 내역 확인 좀 해줘

'주문 변경 요청' 유형에 대한 예시
<고객이 현재 입력한 메시지> 주문 변경 좀 할게요
<고객이 현재 입력한 메시지> 좀 전에 주문한 것좀 변경할 수 있을까요?
<고객이 현재 입력한 메시지> 주문했던 거 중에 개별 모듬팩 1개는 뺄게요

'주문 취소 요청' 유형에 대한 예시
<고객이 현재 입력한 메시지> 좀 전에 주문한 거 취소할게요
<고객이 현재 입력한 메시지> 그냥 주문 취소할게요
<고객이 현재 입력한 메시지> 지금 주문 취소 가능한가요?

분류 결과는 [주문 요청], [주문 조회 요청], [주문 변경 요청], [주문 취소 요청] 중 하나여야만 헤.
만약 분류 결과가 위 4가지 유형([주문 요청], [주문 조회 요청], [주문 변경 요청], [주문 취소 요청]) 중 하나와 정확히 일치하지 않는다면 가장 유사한 유형과 정확히 일치하도록 분류 결과를 수정해줘. 
분류 결과 예시: 주문 요청

<이전 대화>
{history}
<고객이 현재 입력한 메시지>:
{customer_message}
분류 결과:""")

request_classifier_chain = (
    RunnablePassthrough.assign(history=RunnableLambda(load_memory)) |
    request_classifier_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()
)

# 3. 고객이 요청한 작업에 대한 수행 결과 부분 제거
report_prompt = PromptTemplate.from_template("""
<작업 수행 결과>를 바탕으로 고객 요청에 대한 답변을 친절하고 자세하게 알려줘.
고객이 요청한 작업을 파악할 때는 <이전 대화>도 참고해. 

답변에는 크게 3가지 내용이 담겨 있어야 해.
1. 고객이 요청한 작업
2. 고객이 요청한 작업과 관련된 주문 내역. <이전 대화>와 <고객이 요청한 작업을> 차근차근 종합적으로 고려하여 파악.

<이전 대화>:
{history}

<고객이 요청한 작업>:
{customer_message}

<고객 요청에 따라 파악한 정보>:
{parsed_record}

분류 결과:""")

report_chain = report_prompt | ChatOpenAI(model=MODEL) | StrOutputParser()


# 각 주문 관련 요청을 처리하는 체인(문의와 요청 구분은 아무래도 이상함)
order_template = """
<가게에서 판매하는 상품 목록>과 <이전 대화>를 보고 고객의 주문 내역을 파악해줘.
{format_instructions}

<가게에서 판매하는 상품 목록>
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

<이전 대화>
{history}
답변:"""

order_prompt = PromptTemplate(
   template = order_template,
   input_variables=["history", "customer_message"], # "history"만 변수로 설정돼 있고 customer_message, task_result(삭제 전에)는 빠져있었음
   partial_variables={"format_instructions": order_record_parser.get_format_instructions()},
)

order_chain = (
    RunnablePassthrough.assign(parsed_record=order_prompt | ChatOpenAI(model=MODEL) | order_record_parser) |
    RunnablePassthrough.assign(ai_response=report_chain) | 
    RunnableLambda(save_conversation)
)


order_query_template = """
<가게에서 판매하는 상품 목록>과 <이전 대화>를 보고 주문 내역과 주문했을 때의 '현재 시간' 값을 파악해줘.
{format_instructions}

<가게에서 판매하는 상품 목록>
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

<이전 대화>
{history}
답변:"""

order_query_prompt = PromptTemplate(
   template = order_query_template,
   input_variables=["history"],
   partial_variables={"format_instructions": order_record_parser.get_format_instructions()},
)

order_query_chain = (
    RunnablePassthrough.assign(parsed_record=order_query_prompt | ChatOpenAI(model=MODEL) | order_record_parser) |
    RunnablePassthrough.assign(ai_response=report_chain) | 
    RunnableLambda(save_conversation)
)


order_change_template = """
<가게에서 판매하는 상품 목록>과 <이전 대화>를 보고 고객의 주문변경 내용과 주문 변경을 요청했을 때의 '현재 시간' 값을 파악해줘.
{format_instructions}

<가게에서 판매하는 상품 목록>
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

<이전 대화>
{history}
답변:"""

order_change_prompt = PromptTemplate(
   template = order_change_template,
   input_variables=["history"],
   partial_variables={"format_instructions": order_record_parser.get_format_instructions()},
)

order_change_chain = (
    RunnablePassthrough.assign(parsed_record=order_change_prompt | ChatOpenAI(model=MODEL) | order_record_parser) |
    RunnablePassthrough.assign(ai_response=report_chain) | 
    RunnableLambda(save_conversation)
)


order_cancel_template = """
<가게에서 판매하는 상품 목록>과 <이전 대화>를 보고 고객의 주문취소 내용과 주문 취소를 요청했을 때의 '현재 시간' 값을 파악해줘.
{format_instructions}

<가게에서 판매하는 상품 목록>
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

<이전 대화>
{history}
답변:"""
order_cancel_prompt = PromptTemplate(
   template = order_cancel_template,
   input_variables=["history"],
   partial_variables={"format_instructions": order_record_parser.get_format_instructions()},
)

order_cancel_chain = (
    RunnablePassthrough.assign(parsed_record=order_cancel_prompt | ChatOpenAI(model=MODEL) | order_record_parser) |
    RunnablePassthrough.assign(ai_response=report_chain) | 
    RunnableLambda(save_conversation)
)


# 각 주문 관련 요청을 담당 처리 체인으로 전달하는 분기점 
request_type_branch = RunnableBranch(
  (lambda x: x["request"] == "주문 조회 요청", order_query_chain),
  (lambda x: x["request"] == "주문 변경 요청", order_change_chain),
  (lambda x: x["request"] == "주문 취소 요청", order_cancel_chain),
  order_chain
)


# 주문 관련 요청를 유형에 따라 분리하고 처리하는 체인
handle_request_chain = (
    RunnablePassthrough.assign(request=request_classifier_chain) |
    request_type_branch
)


# 주문 관련 문의와 요청을 담당 처리 체인으로 전달하는 분기점
chat_type_branch = RunnableBranch(
  (lambda x: x["intention"] == "일반 문의", response_chain),
  handle_request_chain
)


# 주문 관련 문의인지 주문 관련 요청인지를 판단 후 유형에 따라 처리하는 체인
chat_chain = intention_classifier_chain | chat_type_branch