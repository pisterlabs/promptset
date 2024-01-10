import uuid

from fastapi import APIRouter, Body, Header, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from sqlmodel import select

from .....util.describe import describe
from .....database import SessionDep, RedisConnectionDep
from ....user.util import AuthenticatedUserDep
from .....config import settings
from .....model.message import Message
from .....model.chat import Chat

openai_client = AsyncOpenAI(api_key = settings.openai_api_key)

router = APIRouter(prefix="/next")

@describe(
""" Streams the assistant's response to a user message.

Args:
    user_message (Message):  The message to the assistant.
    x_chat_id (str):  The ID of the chat to get the next message from.
    current_user (AuthenticatedUser):  The user making the request.
    session (Session):  The database session.

Yields:
    MessageChunk:  A chunk of the assistant's response.
""")
@router.post("")
async def next_message(
    *,
    content: str = Body(embed=True),
    x_chat_id: str = Header(),
    current_user: AuthenticatedUserDep,
    session: SessionDep,
    redis_conn: RedisConnectionDep):
    # Create a request ID.  It will be thrown away at the end, but used to
    # identify the request in the redis interrupts.
    request_id = str(uuid.uuid4())
    # Get the chat
    chat = session.exec(select(Chat).where(Chat.id == x_chat_id)).first()
    # Ensure that the user is a member of the chat
    if chat.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat not found")
    # Subscribe to stream interrupt events
    interrupt_channel_name = f'interrupt channel for chat with ID: {chat.id}'
    interrupt_pubsub = redis_conn.pubsub()
    await interrupt_pubsub.subscribe(interrupt_channel_name)
    # Check if the assistant is already responding
    if chat.is_assistant_responding:
        # Interrupt the other stream
        await redis_conn.publish(interrupt_channel_name, request_id)
    # Indicate that the assistant is responding
    session.add(chat)
    chat.is_assistant_responding = True
    session.commit()
    session.refresh(chat)
    # Retrieve the last X messages from the chat, in ascending order of chat_index
    # X = settings.app_chat_history_length
    messages = session.exec(select(Message).where(Message.chat_id == chat.id).order_by(Message.chat_index.desc()).limit(settings.app_chat_history_length-1)).all()
    # Put them in openai format and append the user's message
    openai_messages = [{"role": msg.role, "content": msg.content} for msg in messages][::-1] + [{"role": "user", "content": content}]
    # TODO: Incorporate different AI agents
    response_stream = await openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=openai_messages,
        stream=True,
    )
    async def iter_openai_stream():
        async for chunk in response_stream:
            # Check the interrupt channel for interrupts
            message = await interrupt_pubsub.get_message(ignore_subscribe_messages=True)
            if message and message['data'].decode('utf-8') != request_id:
                # Stop the stream and return early (without removing the is_assistant_responding flag.)
                # The interrupting stream is now respondible for that flag
                yield f'data: \n\n'
                return
            if chunk.choices[0].delta.content is not None:
                yield f'data: {chunk.choices[0].delta.content}\n\n'
        yield f'data: \n\n'
        # Indicate that the assistant is no longer responding
        chat.is_assistant_responding = False
        session.commit()
    return StreamingResponse(iter_openai_stream(), media_type="text/event-stream")