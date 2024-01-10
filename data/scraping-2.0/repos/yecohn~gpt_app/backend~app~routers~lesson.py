from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi import UploadFile, File, Depends
from backend.db.sql.sql_connector import access_sql
from backend.db.sql.tables import User
from backend.db.mongo.mongo_connector import access_mongo
from backend.engine.gpt import GPTClient
from backend.engine.speech_to_text import STT
from backend.engine.text_to_speech import GCPTTS
from backend.app.users.user import UserInfo
import openai
from google.cloud import speech
from ..oauth2 import get_current_user
from backend.app.models import MessageChat
from datetime import datetime 
router = APIRouter()


@router.get("/lesson/", status_code=200)
async def topic_list(
    mongo_db=Depends(access_mongo),
):
    gpt = GPTClient()
    lesson_prompt = mongo_db.find(collection_name="metadata").gett('GPT_metada').get('lesson_prompt')

    lesson = gpt.create_lesson(lesson_prompt)
    res = lesson
    return res