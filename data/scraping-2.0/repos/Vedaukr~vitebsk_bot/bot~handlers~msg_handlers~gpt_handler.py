from bot.bot_instance.bot import bot_instance
from bot.handlers.msg_handlers.shared import get_prompt
from bot.handlers.shared import tg_exception_handler
from services.openai_service import OpenAiService
import telebot

# Singletons
openai_service = OpenAiService()

@bot_instance.message_handler(regexp=r"^(\bgpt\b|\bгпт\b)\s.+")
@tg_exception_handler
def msg_handler(message: telebot.types.Message):
    bot_reply = bot_instance.reply_to(message, "generating...")    
    prompt = get_prompt(message.text)
    openai_response = openai_service.get_response(prompt, str(message.from_user.id))
    bot_instance.edit_message_text(openai_response, message.chat.id, bot_reply.message_id)
