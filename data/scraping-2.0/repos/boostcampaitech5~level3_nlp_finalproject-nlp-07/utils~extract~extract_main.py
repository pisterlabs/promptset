from fastapi import FastAPI
import os
import openai
from pydantic import BaseModel
import configparser
from typing import List, Union, Optional, Dict, Any
import ast

app = FastAPI()

env = os.getenv('MY_APP_ENV', 'local')
config = configparser.ConfigParser()
config.read(f'../../config/config-{env}.ini')
openapi = config['openai']
openai.api_key=openai['api_key']

class Item(BaseModel):
    query: str

def gpt_chat_gen(prompt, model="gpt-3.5-turbo"):
    
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def build_text(input):
    """
    입력 프롬프트 생성
    input: 사용자로 부터 입력받은 문장
    """
    promprt = """
    답변 외에는 다른 출력을 하지마

    음식을 상품명과 조건을 함께 입력할 것이고, 거기서 상품명과 조건을 추출해야돼
    제대로된 문장을 입력하면, 문장에서 '상품명'과 그 상품에 대한 '조건'을 dict 형태로 출력해
    
    아래는 예외의 경우야
    상품명 또는 조건이 둘 중 하나라도 없다고 판단되면 0만 출력해
    또한 상품이 음식이 아니거나, 이상하게 입력하면 0만 출력해
    
    아래는 dict 형태로 출력해야 되는 예시야
    1. 문장 : '맵지 않고 쫄깃한 떡볶이를 추천해줘', 답변 : { '상품' : '떡볶이', '조건' : ['맵지 않고','쫄깃한'] }
    2. 문장 : '냉동삼겹살인데 보관이 편리하고 가성비가 좋은 상품을 추천해줘', 답변 : { '상품' : '냉동삼겹살', '조건' : ['보관이 편리', '가성비가 좋은'] }
    3. 문장 : '신선하고 달고 가격이 적당한 귤', 답변 : { '상품' : '귤', '조건' : ['신선하고', '달고', '가격이 적당한'] }
    4. 문장 : '혼자 먹기 좋은 곱창전골', 답변 : { '상품' : '곱창전골', '조건' : ['혼자 먹기 좋은'] }
    
    아래는 예외의 경우의 예시야
    5. 문장 : '떡볶이', 답변 : 0
    6. 문장 : '혼자먹기 좋고, 바삭한 것 추천해줘', 답변 : 0
    7. 문장 : '튼튼한 핸드폰 거치대를 추천해줘', 답변 : 0
    8. 문장 : 'ㅈㅁ퍼ㅔㅐㅁㅈㅍ', 답변 : 0
    9. 문장 : '자극적인 것', 답변 : 0

    이제 답변을 출력해
    문장 : """
    text = (promprt + input)

    return text

def extract(prompt):
    """
    추출결과 dictionary 형식으로 바꾼 후 예외 처리
    :param prompt: 프롬프트 형식을 지닌 입력 텍스트
    """
    respond = gpt_chat_gen(prompt)
    try:
        tmp = ast.literal_eval(respond)
        if tmp['조건'] == [] or tmp['조건'] == 0:
            #result = 0
            raise HTTPException(status_code=500, detail="chatGPT_invalid-input")
        else:
            result = tmp
    except:
        #result = 0
        raise HTTPException(status_code=500, detail="chatGPT_invalid-input")
    return result

@app.post("/extract")
def get_extract(item: Item):
    input_promprt = build_text(item.query)
    result = extract(input_promprt)
    return result