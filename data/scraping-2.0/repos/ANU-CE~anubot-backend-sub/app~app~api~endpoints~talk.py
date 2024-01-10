import requests

from fastapi import APIRouter, Depends
from fastapi.encoders import jsonable_encoder
from fastapi.responses import ORJSONResponse

import asyncio
import uvicorn

#import models and schema
from app.schema.talk_schema import KakaoAPI
from app.core.config import settings
from app.db.model import Chat
from app.db.connection import get_db

#for vector database   
from qdrant_client import QdrantClient
import openai

#for DB
from sqlalchemy.orm import Session
from sqlalchemy.sql import func
from datetime import datetime, timezone, timedelta

def qdrant_settings():
    openai.api_key = settings.OPENAI_API_KEY

    COLLECTION_NAME = 'anubot-unified'

    qdrant_client = QdrantClient(
        url = settings.QDRANT_URL,
        port= settings.QDRANT_PORT, 
    )


openai.api_key = settings.OPENAI_API_KEY

COLLECTION_NAME = 'anubot-unified'

qdrant_client = QdrantClient(
    url = settings.QDRANT_URL,
    port= settings.QDRANT_PORT, 
)

#qdrant_settings()

def get_recent_chats(id, session: Session):
    chats = session.query(Chat).filter(Chat.id == id).order_by(Chat.datetime.desc()).limit(3).all()
    session.close()
    print (chats)
    return [f'{chat.chat}\n{chat.reply}' for chat in chats] if chats else []

def save_ask(question, final_response, id, session: Session):
    chat = Chat(chat=question, reply=final_response, id=id)
    session.add(chat)
    session.commit()
    session.close()

def build_prompt(question: str, references: list, saved_ask: list) -> str:
    prompt = f"""
    당신은 안동대학교의 궁금한 점을 답변해 주는 챗봇, 아누봇입니다.
    사용자의 질문은 다음과 같습니다.: '{question}'
    당신은 현재 안동대 주변 식당가, 기숙사식, 사무실이나 부서의 전화번호, 장학금에 대한 정보만 알고있으므로, 그 외의 질문은 답변의 정확도가 떨어진다고 통보하세요.
    링크가 있다면 링크의 원문을 그대로 이용하세요.

    자료:
    """.strip()

    references_text = ""

    for i, reference in enumerate(references, start=1):
        text = reference.payload["plain_text"].strip()
        references_text += f"\n{text}"

    prompt += (
        references_text
        + """\n 
        그리고 '사용자:', '아누봇:' 과 같은 표현은 절대 하지 마세요. 매우 중요합니다. 꼭 지키세요.
        """
    )

    prompt += "최근 대화:\n"

    if saved_ask:
        prompt += "다음은 최근 대화입니다."
        prompt += "\n\n".join([f"{i}. {chat}" for i, chat in enumerate(saved_ask, start=1)])
    else:
        prompt += "첫 대화입니다. 질문과 답변을 지어내지 마세요!"
    
    return prompt


async def prompt_ask(question: str, callback_url: str, id: str, session: Session):
    similar_docs = qdrant_client.search(
        collection_name='anubot-unified',
        query_vector=openai.Embedding.create(input=question, model=settings.EMBEDDING_MODEL)["data"][0]["embedding"],
        limit=5,
        append_payload=True,
    )
    saved_ask = get_recent_chats(id, session)

    print('생성중')
    prompt= build_prompt(question, similar_docs, saved_ask)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=400,
        temperature=0.2,
    )
    final_response = response["choices"][0]["message"]["content"]
    responseBody = {
        "version" : "2.0",
        "template" : {
            "outputs" : [
                {
                    "simpleText" : {
                        "text" : final_response
                    }
                }
            ]
        }
    }
    responseCode = requests.post(callback_url, json=responseBody)
    print(final_response)
    print('반환 완료!')
    save_ask(question, final_response, id, session)
    print('저장 완료!')

router = APIRouter()

@router.post(path='/ask', response_model=KakaoAPI, description='API for KakaoTalk Chatbot, with Callback Function')
async def ask(item: KakaoAPI, session: Session = Depends(get_db)):
    try:
        task = asyncio.create_task(prompt_ask(item.userRequest.utterance, item.userRequest.callbackUrl, item.userRequest.user.id, session))
    except requests.exceptions.ReadTimeout:
        pass
    print('반환중')
    return ORJSONResponse({
        "version" : "2.0",
        "useCallback" : True
    })