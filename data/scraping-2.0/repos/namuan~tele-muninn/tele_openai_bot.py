#!/usr/bin/env python3
"""
Listen to messages with tt and ii prefix
If a message begins with tt then it'll send a prompt to OpenAI Completion API
If a message begins with ii then it'll send a prompt to OpenAI Image API
"""
import logging
import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from typing import Optional, Type

import telegram
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    CallbackContext,
    CommandHandler,
    Filters,
    MessageHandler,
    Updater,
)

from common_utils import build_chart_links_for, retry, setup_logging, verified_chat_id
from openai_api import completions, image_creation

load_dotenv()

TELE_MUNINN_OPENAI_BOT = os.getenv("TELE_MUNINN_OPENAI_BOT")


def start(update: Update, _) -> None:
    update.message.reply_text("👋 Enter a prompt after tt (for Text) or ii (for Image)")


def help_command(update: Update, _) -> None:
    update.message.reply_text("Help!")


def generate_report(ticker, update: Update, context: CallbackContext):
    bot = context.bot
    cid = update.effective_chat.id
    update.message.reply_text(f"Looking up #{ticker}", quote=True)

    try:
        daily_chart_link, weekly_chart_link, full_message = build_chart_links_for(ticker)
        bot.send_photo(cid, daily_chart_link)
        bot.send_photo(cid, weekly_chart_link)
        bot.send_message(cid, full_message, disable_web_page_preview=True, parse_mode="Markdown")
    except NameError as e:
        bot.send_message(cid, str(e))


class BaseHandler:
    def __init__(self, text, update: Update, context: CallbackContext):
        self.text = text
        self.update = update
        self.context = context
        self.bot = context.bot
        self.cid = update.effective_chat.id
        update.message.reply_text(f"Processing #{self.text}", quote=True)

    def process(self):
        raise NotImplementedError


class OpenAiText(BaseHandler):
    def process(self):
        prompt_response = completions(self.text)
        self.bot.send_message(self.cid, prompt_response, disable_web_page_preview=True, parse_mode="Markdown")


class OpenAiImage(BaseHandler):
    def process(self):
        image_response = image_creation(self.text)
        if image_response:
            for image in image_response["data"]:
                logging.info("Sending image %s", image)
                self.bot.send_photo(self.cid, image["url"])
        else:
            self.bot.send_message(self.cid, f"No image found for {self.text}")


plain_text_handler_mapping = {
    "tt": OpenAiText,
    "ii": OpenAiImage,
}


def find_plain_text_handler(incoming_text) -> Optional[Type[BaseHandler]]:
    for prefix, handler in plain_text_handler_mapping.items():
        if incoming_text.lower().startswith(prefix):
            return handler

    return None


@retry(telegram.error.TimedOut, tries=3)
def handle_cmd(update: Update, context: CallbackContext) -> None:
    logging.info(f"Incoming update: {update}")
    chat_id = update.effective_chat.id
    if not verified_chat_id(chat_id):
        return
    incoming_message: str = update.message.text
    message_handler_clazz = find_plain_text_handler(incoming_message)
    if message_handler_clazz:
        message_handler = message_handler_clazz(incoming_message, update, context)
        message_handler.process()


def main():
    """Start the bot."""
    logging.info("Starting open-ai bot")
    if not TELE_MUNINN_OPENAI_BOT:
        logging.error("🚫 Please make sure that you set the TELE_MUNINN_OPENAI_BOT environment variable.")
        return False
    else:
        logging.info("🤖 Telegram bot token: %s", TELE_MUNINN_OPENAI_BOT[:5] + "..." + TELE_MUNINN_OPENAI_BOT[-5:])

    updater = Updater(TELE_MUNINN_OPENAI_BOT, use_context=True)

    dispatcher = updater.dispatcher

    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))

    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_cmd))

    updater.start_polling()

    updater.idle()


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        dest="verbose",
        help="Increase verbosity of logging output",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    main()
