from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.callbacks import get_openai_callback
from sqlalchemy.orm import Session

from app import get_list_of_tags, get_notam_messages
from notam_llm import NOTAMLLMChat
from persist.mysql import MySQLEngine
from persist.notam import NOTAM

load_dotenv()

app = FastAPI()
engine = MySQLEngine(echo=False)
notam_llm_chat = NOTAMLLMChat()

notam_tags = get_list_of_tags(engine.get_engine())


@app.get("/get_notam_messages")
async def get_notam_message_list():
    with Session(engine.get_engine()) as session:
        notams = session.query(NOTAM).all()

    return {
        'notam_messages': [
            {
                'notam_id': notam.notam_id,
                'message': notam.message,
                'location': notam.location,
            } for notam in notams
        ]
    }


@app.post("/ask_notam_about")
async def ask_notam_about(notam_ids: List[str]):
    notam_messages = get_notam_messages(engine.get_engine(), notam_ids)
    with get_openai_callback() as callback:
        answer = notam_llm_chat.chat_to_get_notam_about(notam_tags, notam_messages)
    return {
        'format': 'markdown',
        'answer': answer,
        'cost': f'{callback.total_cost} USD',
    }
