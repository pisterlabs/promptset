from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import openai
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# CORS 설정
origins = [
    "http://localhost:3000",  # Add the URL of your React application
    "http://43.202.77.171:3000",  # Address of the React application
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # You can adjust this based on your needs
    allow_headers=["*"],  # You can adjust this based on your needs
)


# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Define prompt
prompt = """Bot: How can I help you?
User: {{$input}}
---------------------------------------------
You are a schedule summary. Please inform the user of this schedule by summarizing it warmly and kindly. 
Also, I want you to give me some advice on the schedule. And send me a lucky comment as a closing comment. Please answer in Korean.
"""

class SummaryResponse(BaseModel):
    summary: str

@app.get("/summarize_node_data/{user_id}")
async def summarize_node_data(user_id: str):
    # Node.js 서버의 엔드포인트 URL
    node_url = f'http://localhost:8000/goal/summary2/{user_id}'  # 사용자 ID에 따라 요청 URL 구성

    try:
        # Node.js 서버에 GET 요청 보내기
        response = requests.get(node_url)

        # 응답 코드 확인
        if response.status_code == 200:
            # 응답 데이터를 가져와서 OpenAI를 사용하여 요약
            node_data = response.text
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"{prompt}\nUser: {node_data}\n",
                max_tokens=2000,
                temperature=0.7,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=None
            )
            summary = response.choices[0].text.strip()
            
            return {"summary": summary}
        else:
            return {"summary": f"오류 응답 코드: {response.status_code}"}

    except Exception as e:
        return {"summary": f"오류 발생: {str(e)}"}
    
    
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
