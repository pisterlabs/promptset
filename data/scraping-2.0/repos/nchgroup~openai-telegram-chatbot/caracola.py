#!/usr/bin/python3

import openai
from telegram.ext import Updater, CommandHandler
from telegram import constants
from functools import wraps
import logging


# Keys and tokens
openai.api_key = "<OPENAI-KEY>" # Set up the OpenAI API key
LIST_OF_ADMINS = [-9999999]  # Telegram Chat Group
TOKEN_BOT = "<TELEGRAM-TOKEN-BOT>" # Token of Telegram Bot
ENGINE = "gpt-4"

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)

logger = logging.getLogger(__name__)


# Protection
def restricted(func):
    @wraps(func)
    def wrapped(update, context, *args, **kwargs):
        user_id = update.message.chat_id
        print(update.message.chat_id)
        if user_id not in LIST_OF_ADMINS:
            print(f"{user_id} not in {LIST_OF_ADMINS}")
            print("Unauthorized access denied for {}:\nmessage: {}.".format(
                str(update.effective_user), str(context.args)))
            return
        return func(update, context, *args, **kwargs)
    return wrapped


# Create a new Telegram bot
updater = Updater(TOKEN_BOT, use_context=True)
dispatcher = updater.dispatcher


@restricted
def handle_message(update, context):
    # Use the OpenAI API to generate a response to the user's message
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=" ".join(context.args),
        max_tokens=1024,
        temperature=0.3,
    )

    message = f"```{response['choices'][0]['text']}\n```"
    max_chars = 4090
    messages = []
    while len(message) > 0:
        if len(message) > max_chars:
            part = message[:max_chars]
            first_space = part.rfind(' ')
            if first_space != -1:
                part = part[:first_space]
                messages.append(part)
                message = message[first_space:]
            else:
                messages.append(part)
                message = message[max_chars:]
        else:
            messages.append(message)
            break

    for m in messages:
        # Send the response back to the user
        context.bot.send_message(
            update.effective_chat.id,
            m,
            parse_mode=constants.PARSEMODE_MARKDOWN_V2,
        )

dp = updater.dispatcher
dp.add_handler(CommandHandler("caracola", handle_message))

# Start the bot
updater.start_polling()

updater.idle()
