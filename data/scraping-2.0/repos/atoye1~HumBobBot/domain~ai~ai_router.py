import os
import openai
from domain.ai import ai_crud


from fastapi import APIRouter, Request

router = APIRouter(
    prefix="/ai",
)

@router.post('/skill')
async def ai_skill(request: Request):
    kakaorequest = await request.json()
    return ai_crud.ai_chat(kakaorequest)

@router.post('/update')
def regulation_update():
    pass
