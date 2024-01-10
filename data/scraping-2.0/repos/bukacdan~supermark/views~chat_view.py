import json
import logging
from typing import AsyncGenerator, Annotated, List

import numpy as np

from fastapi import APIRouter, Header, Query
from langchain.schema import HumanMessage, AIMessage
from starlette.responses import StreamingResponse

from models.bookmark import VectorStoreBookmarkMetadata
from models.chat import ChatServiceMessage, UserSearchMessage, ChatEndpointMessage
from services.chat_history_service import ChatHistoryService
from services.context_service import ContextService
from services.conversation_service import ConversationService
from utils.db import get_vectorstore

router = APIRouter()


logger = logging.getLogger(__name__)

context_service = ContextService(client=get_vectorstore())


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to Python list
        return super(NumpyEncoder, self).default(obj)


async def sse_generator(messages_generator: AsyncGenerator[ChatServiceMessage, None],
                        question: str,
                        conversation_service: ConversationService,
                        conversation_id: str | None = None,
                        chat_history_service: ChatHistoryService | None = None):
    async for msg in messages_generator:
        msg_dict = ChatEndpointMessage(
            chat_response=msg.msg,
            documents=[d.metadata.dict() for d in msg.relevant_documents],
            done=msg.done
        ).dict()
        if msg.done:
            yield f"data: {json.dumps(msg_dict, cls=NumpyEncoder)}\n\n"
            if conversation_id and chat_history_service:
                await chat_history_service.add_chat_message(
                    conversation_id,
                    AIMessage(
                        content=msg.msg,
                    ),
                    used_context=[d for d in msg.relevant_documents]
                )
            else:
                conversation_service.store_conversation(
                    question=question,
                    context=[d for d in msg.relevant_documents],
                    answer=msg.msg,
                )
        else:
            yield f"data: {json.dumps(msg_dict, cls=NumpyEncoder)}\n\n"


@router.get('/chat', responses={200: {"content": {"text/event-stream": {}}}})
async def chat(q: str, conversation_id: str | None = None, selected_context: Annotated[list[str] | None, Query()] = None, x_uid: Annotated[str | None, Header()] = None):
    if not (x_uid):
        raise Exception("user not authenticated")
    conversation_service = ConversationService(context_service=context_service, uid=x_uid)
    chat_history_service = ChatHistoryService(x_uid)
    if conversation_id: # continuous conversation
        await chat_history_service.add_chat_message(
            conversation_id,
            HumanMessage(
                content=q
            )
        )
    completion = conversation_service.chat(
        message=q,
        selected_context=selected_context,
    )
    sse = StreamingResponse(
        sse_generator(completion, q, conversation_service, conversation_id, chat_history_service),
        media_type='text/event-stream'
    )

    # workaround for app engine
    sse.headers["Cache-Control"] = "no-cache"
    return sse


@router.post('/search')
async def search(query: UserSearchMessage, x_uid: Annotated[str, Header()]) -> List[VectorStoreBookmarkMetadata]:
    relevant_docs = context_service.search(
        query.query,
        x_uid,
        certainty=query.certainty,
        alpha=query.alpha,
        limit=query.limit_chunks
    )

    return sorted(list({d.metadata for d in relevant_docs}), key=lambda x: x.similarity_score, reverse=True)

@router.put('/conversation')
async def create_conversation(x_uid: Annotated[str, Header()]):
    conversation_service = ConversationService(context_service=context_service, uid=x_uid)
    conv_id = conversation_service.create_new_conversation()

    return conv_id

@router.get('/conversations')
async def get_conversations(x_uid: Annotated[str, Header()]):
    conversation_service = ChatHistoryService(x_uid)
    conversations = await conversation_service.get_conversations()

    return conversations

@router.get('/chat-history')
async def get_chat_history(conversation_id: str, x_uid: Annotated[str, Header()]):
    chat_history_service = ChatHistoryService(x_uid)
    chat_history = await chat_history_service.get_chat_history(conversation_id)

    return chat_history