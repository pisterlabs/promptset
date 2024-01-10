# This script were written using python.
# Using python-telegram-bot (v20.5) and openai api (chat-completion, gpt-3.5-turbo) libraries.

# Import necessary libraries and modules.
import openai
import os
import logging

from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeVar, Union
from telegram import Update
from telegram._utils.defaultvalue import DEFAULT_TRUE
from telegram._utils.types import DVType
from telegram.ext import filters as filters_module
from telegram.ext._basehandler import BaseHandler
from telegram.ext._utils.types import CCT, HandlerCallback
from telegram.ext import ApplicationBuilder, ContextTypes

# Fetch OPENAI_API_KEY and TELEGRAM_BOT_TOKEN from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')
telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(levelname)s - %(message)s'
)

if TYPE_CHECKING:
    from telegram.ext import Application

RT = TypeVar("RT")

class MessageHandler(BaseHandler[Update, CCT]):

    __slots__ = ("filters",)
 
    def __init__(
        self,
        filters: filters_module.BaseFilter,
        callback: HandlerCallback[Update, CCT, RT],
        block: DVType[bool] = DEFAULT_TRUE,
    ):
        super().__init__(callback, block=block)
        self.filters: filters_module.BaseFilter = (
            filters if filters is not None else filters_module.ALL
        )

    def check_update(self, update: object) -> Optional[Union[bool, Dict[str, List[Any]]]]:
        if isinstance(update, Update):
            return self.filters.check_update(update) or False
        return None


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text
    
    # Make "typing..." status visible under bots name
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    # Use OpenAI API to generate a response
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=f"{user_input}\n",
        max_tokens="4,097"
    )
    # Send the response back to the user
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response["choices"][0]["text"])
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Any more questions?")

if __name__ == "__main__":
    # Create the Telegram bot instance
    application = ApplicationBuilder().token(telegram_bot_token).build()
    
    # Register the message handler
    handler = MessageHandler(filters=None, callback=handle_message)
    application.add_handler(handler)
    # Start the bot
    application.run_polling()
