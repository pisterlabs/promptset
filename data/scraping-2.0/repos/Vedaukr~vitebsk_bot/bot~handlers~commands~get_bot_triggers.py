from bot.bot_instance.bot import bot_instance
from bot.handlers.shared import tg_exception_handler
import telebot
from services.openai_service import OpenAiService
from utils.search_resolver import search_resolver

@bot_instance.message_handler(commands=['get_bot_triggers'])
@tg_exception_handler
def get_bot_triggers(message: telebot.types.Message):
    response = f"Usage: bot [trigger] [trigger_prompt]\n\n"
    response += "Search triggers:\n"
    for handler in search_resolver.handlers:
        uri = handler.get_site_uri()
        response += f"Site: {uri if uri else 'Default search'}, Triggers: {handler.get_triggers()}\n"
    bot_instance.reply_to(message, response)