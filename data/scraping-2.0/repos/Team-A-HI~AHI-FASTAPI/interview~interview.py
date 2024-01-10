from fastapi import FastAPI, APIRouter, Body, File, Request, UploadFile, Depends, Form
import openai
from pydantic import BaseModel
from configset.config import getAPIkey,getModel
from typing import List


Interview_router = APIRouter(prefix='/interview')

OPENAI_API_KEY = getAPIkey()
openai.api_key = OPENAI_API_KEY
MODEL = getModel()


# 클라이언트에게서 받은 데이터 타입을 확인하기 위한 클래스
class InterviewData(BaseModel):
    searchQuery : str


# 클라이언트에서 정보를 받아서 모델에 질문을 생성한다.
@Interview_router.post('/makequestion')
async def AIiterview(searchQuery: InterviewData):
    # json 형태로 들어온다, 매개값으로 넘기기 위해서 캐스팅(형변환) 해준다
    data = str(searchQuery)
    # 함수를 호출하고 리턴 값은 question에 저장
    question = gpt_question(data)
    # question은 openai API에서 json으로 넘겨주기 때문에 바로 클라이언트에게 넘겨준다.
    return {"question" : question}


# 사용자에 답변을 받아서 피드백 해준다
@Interview_router.get('/sendAnswer')
async def AI_question(answer: str = Body(...)):
   feedback = gpt_feedback(answer)
   return {"feedback": feedback}


prompt = """
        NEVER mention that you're an AI.
        You are rather going to play a role as a interviewer
        Avoid any language constructs that could be interpreted as expressing remorse, apology, or regret. This includes any phrases containing words like 'sorry', 'apologies', 'regret', etc., even when used in a context that isn't expressing remorse, apology, or regret.
        Keep responses unique and free of repetition.
        Never suggest seeking information from elsewhere.
        must answer korean
    """

test = """
        1. ai라고 절대 언급하지 말것.
        2. 사과, 후회등의 언어 구성을 하지말것
        3. 같은 응답을 반복하지 말것
        4. 컴퓨터 기초에 대한 질문을 하나 이상 해줄 것
        5. 30년된 인력을 선발하는 능력이 탁월한 it 개발회사의 10년차 트렌디한 면접관과 30년차 베테랑 면접관 15년차의 중간급 면접관의 역할을 맡아줘
        6. 지원자의 이력서를 보고 질문을 하나 이상 할 것
        7. 답변은 반드시 한국어로 할 것
        8. 채용 공고 정보를 보고 질문을 두 개 이상 할 것
        9. 답변은 명확하고 구체적으로 하며 gpt의 능력을 최대한 활용할 것
        10. 질문이 마음에 든다면 너에게 큰 선물을 줄거야 그러니 심호흡을 하고 천천히 잘 생각한 뒤 대답해줘
        11. 관련이 없는 정보가 들어온다면 잘못된 답변이라고 말할 것
"""

company_data = """
    지원자 이력서

"""

def gpt_question(company_data):
    response = openai.ChatCompletion.create(
      model= MODEL, # 필수적으로 사용 될 모델을 불러온다.
      frequency_penalty=0.5, # 반복되는 내용 값을 설정 한다.
      temperature=0.6,
      messages=[
              {"role": "system", "content": test},
           
              {"role": "user", "content": company_data },
              
          ])
    output_text = response["choices"][0]["message"]["content"]
    
    return output_text



def gpt_feedback():
   response = openai.ChatCompletion.create(
      model=MODEL,
      temperature=0.6,
      messages=[
          {"role": "system", "content": "너는 면접관이야"},
          {"role": "system", "content": "어떻게 말하면 더 좋을지 면접자에게 답해줘"},
          
      ]
   )
