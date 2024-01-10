import openai
import telebot

from bot.constant_messages import empty_chat_message, api_key_not_set_message, api_key_expired_message
from bot.constants import bot
from bot.error_handlers import on_api_telegram_exception_429
from db.db import insert_message, insert_user_if_not_exists, insert_group_if_not_exists
from open_ai.chatgpt.reply import get_chatgpt_reply
from open_ai.common import is_openai_api_key_set


def reply_to_text_prompt(message: telebot.types.Message):
    insert_user_if_not_exists(message)
    if message.chat.type == 'group' or message.chat.type == 'supergroup':
        insert_group_if_not_exists(message)
    if is_openai_api_key_set(user_id=message.from_user.id):
        try:
            content = message.text
            if '/chat@sv_telegram_gpt_bot' in content:
                content = content.replace('/chat@sv_telegram_gpt_bot', '')
            if '/chat' in content:
                content = content.replace('/chat', '')
            if not content:
                bot.reply_to(message, empty_chat_message)
                return
            reply = get_chatgpt_reply(
                content=content,
                user_id=message.from_user.id,
                full_name=message.from_user.full_name,
                username=message.from_user.username
            )
            insert_message(message, reply)
            bot.reply_to(message=message, text=reply)
        except telebot.apihelper.ApiTelegramException:
            on_api_telegram_exception_429(reply_to_text_prompt, message)
        except openai.RateLimitError:
            bot.reply_to(message, api_key_expired_message)
    else:
        bot.reply_to(message, api_key_not_set_message('OpenAI'))


def register_chat_text():
    bot.register_message_handler(
        reply_to_text_prompt,
        func=lambda message: message.text.isprintable(),
        chat_types=['group', 'supergroup'],
        commands=['chat']
    )
    bot.register_message_handler(
        reply_to_text_prompt,
        func=lambda message: message.text.isprintable(),
        chat_types=['private']
    )