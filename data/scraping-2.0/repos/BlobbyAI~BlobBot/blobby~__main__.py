import argparse

from telegram import Update
from telegram.ext import filters, CommandHandler, MessageHandler

from blobby import blob_app
from blobby.openai_completions.config import config_argparse
from blobby.openai_completions import OpenAICompletions


if __name__ == "__main__":
    config = config_argparse()
    openai_completions = OpenAICompletions(config)


async def start(update: Update, _) -> None:
    await update.message.reply_text(f"Hi! I'm {config.name} and I respond to your text messages!")


async def blob(update: Update, _) -> None:
    message = update.message
    input_text = message.text
    user = message.from_user
    chat = message.chat

    created_text = openai_completions.create_text(
        input_text,
        chat.id,
        user.id,
        user.username,
    )

    if not created_text:
        return

    await update.message.reply_text(created_text)


def _bot_init() -> None:
    start_handler = CommandHandler("start", start)
    blob_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), blob)
    
    blob_app.add_handler(start_handler)
    blob_app.add_handler(blob_handler)

    blob_app.run_polling(allowed_updates=["message"], drop_pending_updates=True)


if __name__ == "__main__":
    _bot_init()
