from typing import Annotated

from fastapi import APIRouter, Depends
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.language_models import BaseLLM
from langchain_core.memory import BaseMemory
from langchain_core.messages import HumanMessage
from loguru import logger

from pybot.config import settings
from pybot.context import session_id
from pybot.dependencies import ChatMemory, Llm, MessageHistory, UserIdHeader
from pybot.jupyter import ContextAwareKernelManager, GatewayClient
from pybot.models import Conversation as ORMConversation
from pybot.schemas import (
    ChatMessage,
    Conversation,
    ConversationDetail,
    CreateConversation,
    UpdateConversation,
)
from pybot.session import RedisSessionStore, Session
from pybot.summarization import summarize as summarize_conv

router = APIRouter(
    prefix="/api/conversations",
    tags=["conversation"],
)
gateway_client = GatewayClient(host=settings.jupyter_enterprise_gateway_url)
session_store = RedisSessionStore()
kernel_manager = ContextAwareKernelManager(
    gateway_host=settings.jupyter_enterprise_gateway_url
)


@router.get("")
async def get_conversations(
    userid: Annotated[str | None, UserIdHeader()] = None
) -> list[Conversation]:
    convs = await ORMConversation.find(ORMConversation.owner == userid).all()
    convs.sort(key=lambda x: x.updated_at, reverse=True)
    return [Conversation(**conv.dict()) for conv in convs]


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    history: Annotated[RedisChatMessageHistory, Depends(MessageHistory)],
    userid: Annotated[str | None, UserIdHeader()] = None,
) -> ConversationDetail:
    conv = await ORMConversation.get(conversation_id)
    session_id.set(f"{userid}:{conversation_id}")
    return ConversationDetail(
        messages=[
            ChatMessage.from_lc(
                lc_message=message, conv_id=conversation_id, from_=userid
            )
            if isinstance(message, HumanMessage)
            else ChatMessage.from_lc(lc_message=message, conv_id=conversation_id)
            for message in history.messages
        ],
        **conv.dict(),
    )


@router.post("", status_code=201)
async def create_conversation(
    payload: CreateConversation, userid: Annotated[str | None, UserIdHeader()] = None
) -> ConversationDetail:
    conv = ORMConversation(title=payload.title, owner=userid)
    await conv.save()
    # create session
    session = Session(pk=f"{userid}:{conv.pk}", user_id=userid, conv_id=conv.pk)
    await session_store.asave(session)
    session_id.set(session.pk)
    await kernel_manager.astart_kernel()
    return ConversationDetail(**conv.dict())


@router.put("/{conversation_id}")
async def update_conversation(
    conversation_id: str,
    payload: UpdateConversation,
    userid: Annotated[str | None, UserIdHeader()] = None,
) -> None:
    conv = await ORMConversation.get(conversation_id)
    conv.title = payload.title
    await conv.save()


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(
    conversation_id: str,
    userid: Annotated[str | None, UserIdHeader()] = None,
) -> None:
    session = await session_store.aget(f"{userid}:{conversation_id}")
    # delete kernel
    try:
        gateway_client.delete_kernel(str(session.kernel_id))
    except Exception as e:
        logger.exception(f"failed to delete kernel {session.kernel_id}, err: {str(e)}")
    # delete session
    await session_store.adelete(f"{userid}:{conversation_id}")
    # delete conversation
    await ORMConversation.delete(conversation_id)


@router.post("/{conversation_id}/summarization", status_code=201)
async def summarize(
    conversation_id: str,
    llm: Annotated[BaseLLM, Depends(Llm)],
    memory: Annotated[BaseMemory, Depends(ChatMemory)],
    userid: Annotated[str | None, UserIdHeader()] = None,
) -> dict[str, str]:
    session_id.set(f"{userid}:{conversation_id}")
    title = await summarize_conv(llm, memory)
    conv = await ORMConversation.get(conversation_id)
    conv.title = title
    await conv.save()
    return {"title": title}
