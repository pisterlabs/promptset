import uuid

from fastapi import APIRouter
from fastapi.responses import Response
from sqlmodel import select
from openai import AsyncOpenAI
import twilio.twiml.messaging_response as twiml

from ...database import SessionDep, RedisConnectionDep
from ...model.twilio_message import TwilioMessage
from ...model.chat import Chat
from ...model.message import Message
from ...util.describe import describe
from ...config import settings
from .util import TwilioGetUserDep

openai_client = AsyncOpenAI(api_key=settings.openai_api_key)

router = APIRouter(
    prefix="/twilio"
)

@describe(
""" Receives Twilio webhooks

Args:
    request (Request): The request object containing the data sent by Twilio.

Returns:
    TwiML.  The TwiML response.
""")
@router.post("")
async def twilio_webhook(
    incoming_message: TwilioMessage,
    current_user: TwilioGetUserDep,
    session: SessionDep,
    redis_conn: RedisConnectionDep):
    # - Verify that this is not a whatsapp message
    if incoming_message.from_.startswith('whatsapp:'):
        twiml_response = twiml.MessagingResponse()
        twiml_response.message(settings.app_no_whatsapp_message)
        return Response(content=twiml_response.to_xml(),
                        media_type="application/xml")
    # - Verify that this is not a MMS
    if incoming_message.num_media > 0:
        twiml_response = twiml.MessagingResponse()
        twiml_response.message(settings.app_no_mms_message)
        return Response(content=twiml_response.to_xml(),
                        media_type="application/xml")
    # Create a request ID.  It will be thrown away at the end, but used to
    # identify the request in the redis interrupts.
    request_id = str(uuid.uuid4())
    # - If the current_user could not be found, return a static message requesting referral code
    if not current_user:
        twiml_response = twiml.MessagingResponse()
        twiml_response.message(settings.app_request_referral_message)
        return Response(content=twiml_response.to_xml(),
                        media_type="application/xml")
    # - Retrieve the user's twilio chat
    twilio_chat = session.exec(select(Chat).where(Chat.id == current_user.twilio_chat_id)).first()
    # Subscribe to stream interrupt events
    interrupt_channel_name = f'interrupt channel for chat with ID: {twilio_chat.id}'
    interrupt_pubsub = redis_conn.pubsub()
    await interrupt_pubsub.subscribe(interrupt_channel_name)
    # Check if the assistant is already responding
    if twilio_chat.is_assistant_responding:
        # Interrupt the other stream
        await redis_conn.publish(interrupt_channel_name, request_id)
    session.add(twilio_chat)
    twilio_chat.is_assistant_responding = True
    session.commit()
    session.refresh(twilio_chat)
    # Check if this is a new user (twilio chat has no messages yet)
    # If so, send the static welcome message.
    if not twilio_chat.messages:
        twiml_response = twiml.MessagingResponse()
        twiml_response.message(settings.app_welcome_message)
        return Response(content=twiml_response.to_xml(),
                        media_type="application/xml")
    # - Save the incoming message to the database (commit and refresh)
    user_message = Message(
        chat_index=len(twilio_chat.messages),
        chat_id=twilio_chat.id,
        role='user',
        content=incoming_message.body,
    )
    session.add(user_message)
    session.commit()
    session.refresh(user_message)
    # - Begin streaming a response from openAI
    # Create the system and user messages
    openai_system_message = {
        "role": "system",
        "content": settings.app_ai_system_message,
    }
    # Retrieve the last X messages from the chat, in ascending order of chat_index
    # X = settings.app_chat_history_length
    messages = session.exec(select(Message).where(Message.chat_id == twilio_chat.id).order_by(Message.chat_index.desc()).limit(settings.app_chat_history_length-1)).all()
    # Put them in openai format, prepend the system message
    openai_messages = [openai_system_message] + [{"role": msg.role.value, "content": msg.content} for msg in messages][::-1]
    # TODO: Incorporate different AI agents
    response_stream = await openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=openai_messages,
        stream=True,
    )
    # Create the partial_response
    print(f'message with SID: {incoming_message.message_sid}')
    if await redis_conn.get(f'multi-part message with SID: {incoming_message.message_sid}'):
        print("multi-part message detected")
        print(incoming_message.body)
    partial_response = "…" if await redis_conn.get(f'multi-part message with SID: {incoming_message.message_sid}') else ""
    requires_redirect = False
    async for chunk in response_stream:
        # Check the interrupt channel for interrupts
        message = await interrupt_pubsub.get_message(ignore_subscribe_messages=True)
        if message and message['data'].decode('utf-8') != request_id:
            # Stop the stream and return early (without removing the is_assistant_responding flag.)
            # The interrupting stream is now respondible for that flag
            # Return no message or redirect.  The interrupting stream is in control.
            return twiml.MessagingResponse()
        partial_response += chunk.choices[0].delta.content or ""
        # Check if the partial response is over the character limit
        if len(partial_response) > settings.app_max_sms_characters:
            # Break at the character limit, add an ellipsis emoji, and stop streaming.  Return the message.
            partial_response = partial_response[:settings.app_max_sms_characters-1] + "…"
            requires_redirect = True
            await redis_conn.set(f'multi-part message with SID: {incoming_message.message_sid}', 'True', ex=60)
            break
        # Check if the partial response has a message break
        if "|" in partial_response:
            # Break at that message and stop streaming.  Return the message.
            partial_response = partial_response.split("|")[0]
            requires_redirect = True
            break
    # - Save the AI message to the database (commit and refresh)
    ai_message = Message(
        chat_index=len(twilio_chat.messages),
        chat_id=twilio_chat.id,
        role='assistant',
        content=partial_response,
    )
    session.add(ai_message)
    session.commit()
    session.refresh(ai_message)
    # - Send the response to Twilio, with a redirect if the stream was incomplete.
    twiml_response = twiml.MessagingResponse()
    twiml_response.message(ai_message.content)
    if requires_redirect:
        twiml_response.redirect('./')
    return Response(content=twiml_response.to_xml(),
                    media_type="application/xml")
    # TODO: Think about how to identify a redirect, so a user's message is not repeated.