import json
import logging
from typing import AsyncGenerator

import numpy as np

from fastapi import APIRouter
from langchain.embeddings import OpenAIEmbeddings
from starlette.responses import StreamingResponse

from models.chat import UserChatMessage, ChatServiceMessage, UserSearchMessage
from services.conversation_service import ConversationService
from utils.db import get_vectorstore

router = APIRouter()


logger = logging.getLogger(__name__)



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        return super(NumpyEncoder, self).default(obj)


async def sse_generator(messages_generator: AsyncGenerator[ChatServiceMessage, None]):
    async for msg in messages_generator:
        msg_dict = {'chat_response': msg.msg, 'documents': [d.dict() for d in msg.relevant_documents], 'done': msg.done}
        if msg.done:
            # save to db
            print(f'yielding {msg.dict()}')
            yield f"data: {json.dumps(msg_dict, cls=NumpyEncoder)}\n\n"
        else:
            print(f'yielding {msg.dict()}')
            yield f"data: {json.dumps(msg_dict, cls=NumpyEncoder)}\n\n"


@router.get('/chat', responses={200: {"content": {"text/event-stream": {}}}})
async def chat(q: str):
    chat_service = ConversationService(vectorstore=get_vectorstore('text', OpenAIEmbeddings()))
    completion = chat_service.chat(
        message=q,
    )
    sse = StreamingResponse(sse_generator(completion),
                            media_type='text/event-stream')

    # workaround for app engine
    sse.headers["Cache-Control"] = "no-cache"
    return sse


@router.post('/search')
async def search(message: UserSearchMessage):
    chat_service = ConversationService(vectorstore=get_vectorstore('text', OpenAIEmbeddings()))
    relevant_docs = chat_service.get_context(message.query)
    return [d.dict() for d in relevant_docs]