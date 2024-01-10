import openai
import os
from fastapi import APIRouter, Depends, HTTPException
from Service.TDX import getData
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import Service.Token as Token

router = APIRouter(tags=["外部服務(Dev Only)"],prefix="/Service/Email")

@router.get("/ChatGPT", summary="OpenAI NLP處理")
async def ChatGPT(message : str,require: str,token: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    """
    require： 對輸出的要求
    message：傳入訊息
    """
    openai.api_key = app_id = os.getenv('OpenAI_Key')
    user = message + require
    if user:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "我需要用繁體中文輸出"},
                {"role": "user", "content": user},
            ]
        )
        return (response['choices'][0]['message']['content'])
    

def chatGPT(message : str,require: str,token: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    openai.api_key = app_id = os.getenv('OpenAI_Key')
    user = message + require
    if user:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "我需要用繁體中文輸出"},
                {"role": "user", "content": user},
            ]
        )
        return (response['choices'][0]['message']['content'])
