from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import json
import uuid
from fastapi import File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Integer, Date
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
import requests
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import io
from PIL import Image

# api key 불러오기
with open("secrets.json") as config_file:
    config_data = json.load(config_file)
    print(config_data)
OPENAI_API_KEY = config_data["openai_api_key"]
HUG_API_KEY=config_data["hug_api_key"]

# # 이미지 파일 저장 경로
IMAGE_STORAGE_PATH = "/home/ubuntu/img/"
#IMAGE_STORAGE_PATH="/Users/ss3un9/fastapi/test_images/"


openai.api_key = OPENAI_API_KEY


app = FastAPI()

with open("db.json") as secret_file:
    secret_data = json.load(secret_file)
    db_user = secret_data["db_user"]
    db_password = secret_data["db_password"]
    db_host = secret_data["db_host"]
    db_name = secret_data["db_name"]

# MySQL 연결 문자열 생성
SQLALCHEMY_DATABASE_URL = f"mysql://{db_user}:{db_password}@{db_host}/{db_name}"
# 데이터베이스 엔진 및 세션 설정
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Data(Base):
    __tablename__ = "data"

    no = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)  # 날짜 컬럼 추가
    title = Column(String)
    weather = Column(String)
    contents = Column(String)
    img_location = Column(String)

#CORS
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # cookie 포함 여부를 설정한다. 기본은 False
    allow_methods=["*"],    # 허용할 method를 설정할 수 있으며, 기본값은 'GET'이다.
    allow_headers=["*"],	# 허용할 http header 목록을 설정할 수 있으며 Content-Type, Accept, Accept-Language, Content-Language은 항상 허용된다.
)

class QuestionInput(BaseModel):
    title: str
    weather: str
    contents: str

@app.post("/posts/save")
async def get_answer(data: QuestionInput):
    title=data.title
    weather=data.weather
    contents=data.contents

    # GPT-3 호출 (v1/chat/completions 엔드포인트 사용)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are the assistant summarizing the sentence"},
            {"role": "user", "content":  f"{contents} \n Please translate the previous sentence into English"},
        ],
        
        temperature=0.5,
    )
    
    answer = response.choices[0].message["content"].strip()
    

    print(f"prompt : {answer}") #프롬프트 출력 

    API_URL = "https://api-inference.huggingface.co/models/nerijs/pixel-art-xl" # 사용 모델 
    headers = {"Authorization": HUG_API_KEY} 
    object_key_name = str(uuid.uuid4()) + '.jpg'  # 랜덤이름

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    
    image_bytes = query({
        "inputs": answer, 
    })

    image = Image.open(io.BytesIO(image_bytes))

    temp_file = IMAGE_STORAGE_PATH+object_key_name # 이미지를 저장할 경로 설정
    image.save(temp_file)
    

    current_date = datetime.now().strftime("%Y-%m-%d") 


    db = SessionLocal()
    db_entry = Data(title=title,date=current_date,weather=weather, contents=contents, img_location=object_key_name)
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    db.close()

    return {"image_name": f"static/{object_key_name}"} 



@app.get("/posts/result/{filename}")
async def get_result(filename: str):
    # 데이터베이스에서 filename과 일치하는 행 조회
    img_name=filename+".jpg"
    db = SessionLocal()
    data_entry = db.query(Data).filter_by(img_location=img_name).first()
    db.close()

    if data_entry:
        # 조회된 데이터가 있는 경우 반환
        return {
            "date": data_entry.date.strftime("%Y-%m-%d"),
            "title": data_entry.title,
            "weather": data_entry.weather,
            "contents": data_entry.contents,
            "img_name": "static/" + data_entry.img_location,
        }
    else:
        # 데이터가 없는 경우 에러 반환
        raise HTTPException(status_code=404, detail="Data not found")