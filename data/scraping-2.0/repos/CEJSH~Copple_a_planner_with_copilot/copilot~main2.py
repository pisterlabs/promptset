from fastapi import FastAPI
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv

# .env 파일로부터 환경 변수 로드
load_dotenv()

app = FastAPI()

# 환경 변수에서 OpenAI API 키 가져오기
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# 프롬프트 정의
prompt1 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are a classifier that categorizes the input as either a goal, an event, or a to-do:
Goal: Refers to a result or state that one aims to achieve within a specific time frame or an undefined period. Goals can be short-term or long-term, and they can be personal or related to a group or organization.
Event: A happening or occasion that takes place at a specific time and location. The time is specifically set on a daily or hourly basis.
To-Do: Refers to a small task or duty that needs to be accomplished.
When answering, please only answer classification.

"""

# 프롬프트2 정의
prompt2 = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are an action type recognizer that categorizes the input as either a create, read, update, or delete:
Create: Includes the act of meeting someone or doing something.
Read: Refers to the act of consuming information or data.
Update: Involves modifying or changing existing information or data.
Delete: Involves removing or discarding something.
When answering, please only answer the type of action.
"""

# 입력을 분류하는 함수
def get_intent(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt1}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()

# 입력을 분류하는 함수 (플랜 추가 관련)
def get_plan_intent(input_text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{prompt2}\nUser: {input_text}\n",
        max_tokens=500,
        temperature=0.7,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=None
    )
    return response.choices[0].text.strip()

# 요청과 응답을 위한 Pydantic 모델 정의
class InputRequest(BaseModel):
    input: str

class ClassificationResponse(BaseModel):
    classification: str

# 분류를 위한 엔드포인트 생성
@app.post("/plan_type")
async def plan_type(input_request: InputRequest):
    input_text = input_request.input
    result = get_intent(input_text)
    return {"classification": result}

# 추가된 엔드포인트 (플랜 추가 관련)
@app.post("/plan_crud")
async def plan_crud(input_request: InputRequest):
    input_text = input_request.input
    result = get_plan_intent(input_text)
    return {"classification": result}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)