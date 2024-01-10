from telegram import Update
from telegram.ext import ContextTypes

from openai_manager.manager import get_answer


async def start_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Ask any question")


async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    answer = get_answer(question=update.message.text)
    await context.bot.send_message(chat_id=update.effective_chat.id, text=answer)
