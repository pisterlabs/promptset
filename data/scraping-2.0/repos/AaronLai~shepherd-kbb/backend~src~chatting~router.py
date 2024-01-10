from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile

from backend.src.chatting.service import ChattingService
from backend.src.chatting.model import ChattingDTO

from backend.config import Settings, get_settings
from backend.src.utilities.openai import OpenAi
import pinecone



router = APIRouter()
settings = get_settings()
#load embeddings from openai
embeddings = OpenAi(settings.OPENAI_API_KEY).getEmbeddings();
pinecone.init(
                api_key=settings.PINECONE_API_KEY, 
                environment=settings.PINECONE_ENVIRONMENT 
            )



@router.post('/')
async def upload(chatting: ChattingDTO, settings: Settings = Depends(get_settings)):
    chattingService = ChattingService(settings)
    result = chattingService.chat_with_namespace(settings, embeddings , chatting.projectId ,   chatting.text , chatting.history)
    return {
        'answer':result['text'],
        'source':result['source']
    }
    
