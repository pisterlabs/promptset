import os

from open_ai_client import OpenAiClient
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from models.chat import Chat


class TelegramBot:
    def __init__(self, openai: OpenAiClient):
        self.openai = openai

        self.application = ApplicationBuilder().token(os.environ['TELEGRAM_TOKEN']).build()
        self.application.add_handler(CommandHandler('ask', self._ask_command_async))
        self.application.add_error_handler(self.error_handler_context)

    def run(self):
        self.application.run_polling()

    async def _ask_command_async(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat = Chat(update.effective_chat.id, context.bot, self.openai)
        return await chat.send_message_async(update.message.text)

    async def error_handler_context(self, update, context):
        print(update.error)
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                       text="Something went wrong. Please, try again")


