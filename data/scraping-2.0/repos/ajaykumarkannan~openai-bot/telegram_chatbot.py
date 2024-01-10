#!/usr/bin/env python3
import os
import openai
import random
from telegram import Update
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)
from telegram import InputMediaPhoto, ParseMode, ChatAction
import time
import logging
from stable_diffusion import sd_call

from keys import openai_api_key
from keys import telegram_bot_key

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def help(update: Update, _: CallbackContext) -> None:
    update.message.reply_text(
        "Help Menu:\n"
        "/prompt <text> - Use a query\n"
        "/p <text> - Use a query\n"
        "/s <text> - build an image using stable diffusion\n"
        "/help - Print this menu\n"
        "/info - to print the README"
    )


def info(update: Update, _: CallbackContext) -> None:
    readme = open("README.md", "r")
    outString = readme.read()
    update.message.reply_text(outString)
    readme.close()


def prompt(update: Update, context: CallbackContext) -> None:
    try:
        openai.api_key = openai_api_key
        model = "text-davinci-003"
        max_token_value = 3000

        update.message.reply_chat_action(action=ChatAction.TYPING)

        if len(context.args) == 1:
            prompt_value = context.args[0]
        elif len(context.args) < 2:
            raise KeyError("Type something in, you dumb POS")
        else:
            prompt_value = " ".join(context.args[:])

        # create a completion
        completion = openai.Completion.create(
            engine=model, prompt=prompt_value, max_tokens=max_token_value
        )

        # print the completion
        update.message.reply_text((completion.choices[0].text))
    except KeyError:
        update.message.reply_text(KeyError)


def stable_diffusion(update: Update, context: CallbackContext) -> None:
    try:
        if len(context.args) == 1:
            prompt_value = context.args[0]
        elif len(context.args) < 2:
            raise KeyError("Type something in, you dumb POS")
        else:
            prompt_value = " ".join(context.args[:])

        update.message.reply_chat_action(action=ChatAction.UPLOAD_PHOTO)
        image_num = random.randint(1000, 9999)
        sd_call(prompt_value, image_num)
        filename = "output" + str(image_num) + ".png"

        update.message.reply_photo(open(filename, "rb"))
        os.remove(filename)
    except KeyError:
        update.message.reply_text(KeyError)


def main() -> None:
    """Run bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater(telegram_bot_key)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    function_list = {
        "start": help,
        "prompt": prompt,
        "p": prompt,
        "s": stable_diffusion,
        "info": info,
    }
    for key, value in function_list.items():
        dispatcher.add_handler(CommandHandler(key, value))

    # Start the Bot
    updater.start_polling()

    # Block until you press Ctrl-C or the process receives SIGINT, SIGTERM or
    # SIGABRT. This should be used most of the time, since start_polling() is
    # non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == "__main__":
    main()
