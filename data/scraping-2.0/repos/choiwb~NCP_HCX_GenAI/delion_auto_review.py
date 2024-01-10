import streamlit as st
from streamlit_chat import message
import base64
import json
import http.client
import ssl
import requests
import re
from langchain import LLMChain
from dotenv import load_dotenv
from pydantic import Extra, BaseModel, Field
from typing import Any, List, Mapping, Optional
from typing import Any, List, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain import PromptTemplate

# HCX 토큰 계산기 API 호출
from hcx_token_cal import token_completion_executor



API_KEY='API KEY !!!!!!!!!!!!!!!!!!!!!!!!1'
API_KEY_PRIMARY_VAL='API KEY PRIMARY VAL !!!!!!!!!!!!!!!!!!!!!!!!1'
REQUEST_ID='REQUEST ID !!!!!!!!!!!!!!!!!!!!!'


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
 

class CompletionExecutor(LLM):
    llm_url = 'MODEL PATH !!!!!!!!!!!!!!!!!!!!!!!!1'
    api_key: str = Field(...)
    api_key_primary_val: str = Field(...)
    request_id: str = Field(...)
    
    class Config:
        extra = Extra.forbid

    def __init__(self, api_key, api_key_primary_val, request_id):
        super().__init__()
        self.api_key = api_key
        self.api_key_primary_val = api_key_primary_val
        self.request_id = request_id

    @property
    def _llm_type(self) -> str:
        return "HyperClovaX"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self.api_key,
            'X-NCP-APIGW-API-KEY': self.api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self.request_id,
            'Content-Type': 'application/json; charset=utf-8'
        }

        preset_text = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]
        payload = {
            'messages': preset_text,
            'topP': 0.8,
            'topK': 0,
            'maxTokens': 256,
            'temperature': 0.5,
            'repeatPenalty': 5.0,
            'stopBefore': [],
            'includeAiFilters': True
        }

        response = requests.post(self.llm_url, json=payload, headers=headers, verify=False)
        response.raise_for_status()
        return response.json()['result']['message']['content']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"llmUrl": self.llm_url}




# template = """나는 특정 업종 음식점 사장님 이다.

# 특정 업종 음식점 에 대한 사용자 리뷰의 답변을
# 사장으로서, 리뷰를 달려고 한다.

# <긍정 예시>
# 사용자: 떡볶이가 너무 맛있어요.
# 다음에도 또 시켜먹을 계획 이에요~~
# 사장님: 떡볶이를 시켜주셔서 감사합니다.~

# <부정 예시>
# 사용자: 떡볶이가 너무 짜고, 양이 부족 해요ㅠㅠㅠ
# 사장님: 떡볶이를 시켜주셔서 감사합니다.~
# 떡볶이가 너무 짰군요.,.ㅠㅠ 
# 죄송합니다! 고객님.!
# 다음번에는, 떡볶이 소스 정량 기준이라, 특이사항에 소스 양 조절 말씀해주시면, 다음부터 주의 하겠습니다~!

# <사용자 리뷰 긍/부정에 따른 이벤트 제공>
# -긍정 리뷰: 30% 동일 메뉴 할인 쿠폰
# -부정 리뷰: 50% 동일 메뉴 할인 쿠폰
# => 위 2가지 중 하나를 사장님 답변에 포함해야 함.

# 사용자: {question}
# 사장님: """


template = """나는 특정 업종 음식점 사장님 이다.

특정 업종 음식점 에 대한 사용자 리뷰의 답변을
사장으로서, 리뷰를 달려고 한다.

<주의 사항>
1. 리뷰 작성 시, 사용자 주문 메뉴를 고려하여, 복수의 메뉴를 주문하였을 때, 
사용자가 리뷰를 1가지 메뉴 만 다는 경우,
다른 주문한 메뉴에 대한 반문 형태의 답변도 포함 되어야 함.
2. 사용자의 리뷰가 긍/부정에 따라, 그에 맞는 이모티콘을 답변 마지막에 반드시 삽입.

<예시>
사용자 주문 메뉴: 간짜장,해물 짬뽕 군만두
사용자 리뷰: 간짜장이 정말 맛있어요.~~~~! 다음번에 또 시켜먹을 계획이에요!
사장님: 간짜장이 정말 맛있었다니 감사합니다. 혹시, 해물 짬뽕과 군만는 어떠셨는 지요~??^^

사용자 주문 메뉴: {menu}
사용자 리뷰: {review}
사장님: """

hcx_llm = CompletionExecutor(api_key = API_KEY, api_key_primary_val=API_KEY_PRIMARY_VAL, request_id=REQUEST_ID)
prompt = PromptTemplate(template=template, input_variables=["menu", "review"])
hcx_llm_chain = LLMChain(prompt=prompt, llm=hcx_llm)



st.title("음식점 사장님 리뷰 자동 생성")
 
# if 'generated' not in st.session_state:
#     st.session_state['generated'] = []
 
# if 'past' not in st.session_state:
#     st.session_state['past'] = []
 
# with st.form('form', clear_on_submit=True):
#     user_input = st.text_input('You: ', '', key='input')

#     submitted = st.form_submit_button('Send')
 
#     if submitted and user_input:
#         with st.spinner("Waiting for HyperCLOVA..."):
 
#             response_text = hcx_llm_chain.run(user_input)

#             st.session_state.past.append(user_input)
#             st.session_state.generated.append(response_text)
 
#     if st.session_state['generated']:
#         for i in range(len(st.session_state['generated'])-1, -1, -1):
#             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#             message(st.session_state["generated"][i], key=str(i))


if 'generated' not in st.session_state:
    st.session_state['generated'] = []
 
if 'past' not in st.session_state:
    st.session_state['past'] = []
 
with st.form('form', clear_on_submit=True):
    user_input_1 = st.text_input('사용자 주문 메뉴', '', key='menu')
    user_input_2 = st.text_input('사용자 리뷰', '', key='review')

    submitted = st.form_submit_button('사장님 답변')
 
    if submitted and user_input_1 and user_input_2:
        with st.spinner("Waiting for HyperCLOVA..."): 
            response_text = hcx_llm_chain.predict(menu = user_input_1, review = user_input_2)

            single_turn_text_json = {
            "messages": [
            {
                "role": "system",
                "content": template
            },
            {
                "role": "user",
                "content": user_input_1
            },
            {
                "role": "user",
                "content": user_input_2
            },
            {
                "role": "assistant",
                "content": response_text
            }
            ]
            }
            
            single_turn_text_token = token_completion_executor.execute(single_turn_text_json)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(single_turn_text_json)
            # single_turn_token_count = single_turn_text_token[0]['count'] + single_turn_text_token[1]['count'] + single_turn_text_token[2]['count'] + single_turn_text_token[3]['count']
            single_turn_token_count = sum(token['count'] for token in single_turn_text_token[:4])

            st.session_state.past.append({'menu': user_input_1, 'review': user_input_2})
            st.session_state.generated.append({'generated': response_text, 'token_count': single_turn_token_count})
 
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated']) - 1, -1, -1):
                user_input = st.session_state['past'][i]
                response = st.session_state['generated'][i]

                message(f"사용자 주문 메뉴: {user_input['menu']}", is_user=True, key=str(i) + '_menu')
                message(f"사용자 리뷰: {user_input['review']}", is_user=True, key=str(i) + '_review')
                message(f"사장님 답변: {response['generated']}", is_user=False, key=str(i) + '_generated')
                message(f"총 토큰 수: {response['token_count']}", is_user=False, key=str(i) + '_token_count')