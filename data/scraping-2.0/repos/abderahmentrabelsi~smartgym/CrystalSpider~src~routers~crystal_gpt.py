import asyncio
import time
import uuid
from typing import Dict

from fastapi import APIRouter, Cookie
from fastapi.responses import JSONResponse
from langchain.memory import ConversationBufferMemory

from crystal_gpt.engine import Engine
from loguru import logger

from crystal_gpt.prompts import PromptController
from fastapi_events.handlers.local import local_handler
from fastapi_events.typing import Event

router = APIRouter(prefix="/crystal_gpt", tags=["Crystal GPT"])
engine = Engine()
session_memories: Dict[str, ConversationBufferMemory] = {}
session_last_access: Dict[str, float] = {}


@local_handler.register(event_name="new-fulltext-added")
def handle_new_document_events(event: Event):
    event_name, payload = event
    logger.info(f"Received event: {event_name} with payload: {payload.id}")
    engine.index_manager.add_article(payload)
    engine.reticulate_splines()


def get_chat_session(memory_key: str = Cookie(None)) -> tuple[str, ConversationBufferMemory]:
    if not memory_key:
        memory_key = str(uuid.uuid4())
    if memory_key not in session_memories:
        session_memories[memory_key] = PromptController.init_conversation_memory()
        session_last_access[memory_key] = time.time()
        logger.debug(f"New session created with key: {memory_key}")

    return memory_key, session_memories[memory_key]


def delete_chat_session(session_key: str) -> None:
    if session_key in session_memories:
        del session_memories[session_key]
        del session_last_access[session_key]
        logger.debug(f"Deleted memory for session: {session_key}")


@router.get("/ask")
async def query(user_query: str, memory_key: str = Cookie(None), articles_only: bool = False, agent: str = None):
    key, memory = get_chat_session(memory_key)
    session_last_access[key] = time.time()
    logger.debug(f"Session '{key}' last access updated")

    chat_response = await engine.query(user_query, memory, articles_only=articles_only, agent=agent)

    response = JSONResponse(content={"response": chat_response.strip()})
    if not memory_key:
        response.set_cookie(key="memory_key", value=key, httponly=True)

    return response


async def session_cleanup_task(session_lifetime: int = 3600):
    while True:
        await asyncio.sleep(session_lifetime)
        current_time = time.time()
        expired_sessions = []
        for session_key, last_access in session_last_access.items():
            if current_time - last_access > session_lifetime:
                expired_sessions.append(session_key)
        for session_key in expired_sessions:
            delete_chat_session(session_key)
            logger.debug(f"Removed memory for expired session: {session_key}")


@router.on_event("startup")
async def on_startup():
    asyncio.create_task(session_cleanup_task())
