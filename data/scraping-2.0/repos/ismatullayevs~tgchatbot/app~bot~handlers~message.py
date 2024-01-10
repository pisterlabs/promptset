from aiogram import types, Router, F
from aiogram.enums import ParseMode
from app.models.user import User
from app.models.message import Message
from app.utils import get_history
from openai import OpenAI
from config.settings import settings

message_handler = Router()
client = OpenAI(api_key=settings.OPENAI_API_KEY)


@message_handler.message(F.text)
async def prompt(message: types.Message):
    assert message.from_user is not None
    user = await User.get(id=message.from_user.id).select_related('last_message')

    reply_id = message.reply_to_message.message_id if message.reply_to_message else user.last_message.message_id if user.last_message else None
    reply = await Message.get_or_none(message_id=reply_id, chat_id=message.chat.id)
    history = await get_history(reply, message.chat.id) if reply else []
    history.append({"role": "user", "content": message.text})

    response = client.chat.completions.create(
        messages=history,
        model='gpt-3.5-turbo',
    )
    response_text = response.choices[0].message.content
    response_msg = await message.answer(str(response_text), parse_mode=ParseMode.MARKDOWN)

    msg = await Message.create(message_id=message.message_id, chat_id=message.chat.id, 
                         text=message.text, role='user', replied_to=reply)
    
    resp = await Message.create(message_id=response_msg.message_id, chat_id=message.chat.id,
                         text=response_text, role='assistant', replied_to=msg)
    
    user.last_message = resp
    await user.save()
    