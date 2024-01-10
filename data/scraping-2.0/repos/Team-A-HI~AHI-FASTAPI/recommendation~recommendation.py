from fastapi import APIRouter , Request , Form , HTTPException , FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import chromadb
from chromadb.db.base import UniqueConstraintError
from sentence_transformers import SentenceTransformer
import openai
from typing import List, Optional
from fastapi.responses import JSONResponse
from datetime import datetime
from configset.config import getAPIkey,getModel
import os
import uuid
import base64
import pdfplumber
from io import BytesIO



OPENAI_API_KEY = getAPIkey()
openai.api_key = OPENAI_API_KEY
MODEL = getModel()
UPLOAD_DIR = "recommendation/resumeImg"

RErouter = APIRouter(prefix="/recommendation")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_question(data):
    response = openai.ChatCompletion.create(
        model= MODEL, # 필수적으로 사용 될 모델을 불러온다.
        frequency_penalty=0.5, # 반복되는 내용 값을 설정 한다.
        temperature=0.6,
        messages=[
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "{data}의 이력서를 읽어"},
            {"type": "text", "text": "회사 이름 옆에 괄호에 팀명이있어 이걸 IT 기업 인지 판단해서 IT 기업에서 일한 기간이 없으면 신입, 1년이상이면 1년이상, 3년이상이면 3년이상, 5년이상이면 5년이상이라고 해줘"},
            {"type": "text", "text": "최종 학력이 대학원 박사졸업했으면 대학원 박사졸업, 대학원 석사졸업했으면 대학원 석사졸업 \
                대학교 4년 다녔으면 대학졸업4년, 대학졸업(2년,3년)은 대학졸업2년,3년, 고등학교 졸업은 고등학교 졸업이라고 알려줘  "},
            
        #     {
        #         "type": "image_url",
        #         "image_url": {
        #             "url": f"data:image/jpeg;base64,{data}",
        #             "detail" : "high"
        #   },
        # },
      ],
    }
  ],
    max_tokens=1000,
    )
    output_text = response["choices"][0]["message"]["content"]
    
    return output_text


@RErouter.post("/resume")
async def get_posting(file: UploadFile = File(...)):
    try:
        # 파일을 비동기 방식으로 읽고 동기 방식으로 얻기
        content = await file.read()

        filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
        file_path = os.path.join(UPLOAD_DIR, filename)

        with pdfplumber.open(BytesIO(content)) as pdf:
            # 각 페이지의 텍스트를 추출하여 리스트로 저장
            text_content = [page.extract_text() for page in pdf.pages]

        print(text_content)

        with open(file_path, "wb") as fp:
            fp.write(content)

        base64_image = encode_image(file_path)

        question = gpt_question(text_content)

        print(question)
        # 결과를 반환 (저장된 파일의 경로를 추가)
        return {"success": True, "file_path": file_path}

    except Exception as e:
        # 예외 처리
        return {"error": str(e)}

