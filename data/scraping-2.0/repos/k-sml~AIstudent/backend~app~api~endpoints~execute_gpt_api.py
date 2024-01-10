from fastapi import APIRouter
from crud.topic_crud import *
from crud.answer_crud import *
from schemas.execute_gpt_api_schema import Response
from typing import List
import os
from schemas.topic_schema import Topic
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

@router.post("/execute/new/{topic_id}", tags=["Execute"])
def first_execute_gpt_api(topic:Topic):
    # topic = select_topic(topic_id)
    client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
    res = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {"role": "system", "content": topic.first_header},
            {"role": "user", "content": topic.first_prompt},
        ]
    )
    return res

@router.post("/execute/{answer_id}", tags=["Execute"])
def execute_openai_api(answer_id: str, res: List[Response]):
    answer = get_answer(answer_id)
    client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
    res.append(
        {"role": "user", "content": answer.content}
    )
    
    res = client.chat.completions.create(
        model = "gpt-4",
        messages = res
    )
    return res

@router.post("/execute/eval/{topic_id}", tags=["Execute"])
def execute_evaluation(res: List[Response]):
    client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
    res.append(
        {"role": "system", "content": "ここまでのやりとりを踏まえて相手のトピックに対する理解度を0%から100%の間で決めてください。%を抜いた数値を出力してください。"}
    )
    
    res = client.chat.completions.create(
        model = "gpt-4",
        messages = res
    )
    return res