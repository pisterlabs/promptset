from bot.bot_instance.bot import bot_instance
from bot.handlers.shared import tg_exception_handler
import telebot
from services.openai_service import OpenAiService

openai_service = OpenAiService()

@bot_instance.message_handler(commands=['get_gpt_context'])
@tg_exception_handler
def get_ctx_handler(message: telebot.types.Message):
    ctx = openai_service.get_or_create_context(str(message.from_user.id))
    bot_instance.reply_to(message, f"Your context:\n{ctx if ctx else 'empty ctx'}")