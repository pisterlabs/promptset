import logging

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import os
import openai

import requests
from bs4 import BeautifulSoup
import re

openai.api_key = ""
# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Welcome! Our command:\n/help - to see all commands\n/getlightinf - search light information\n/compresstxt - to minimize text\n/findartical - to find needed')


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('/getlightinf - search light information\n/compresstxt - to minimize text\n/findartical - to find needed')


def getlightinf(update, context):
    """Send a message when the command /getlightinf is issued."""
    gpt_prompt = update.message.text

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.3,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    
    update.message.reply_text(response['choices'][0]['text'])


def compresstxt(update, context):
    """Send a message when the command /compresstxt is issued."""
    gpt_prompt = "Correct this to standard English:\n\n" + update.message.text

    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=175,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    update.message.reply_text(response['choices'][0]['text'])


def findartical(update, context):
    """Send a message when the command /findarticle is issued."""
    def news(href):
        return href and re.compile("/en/").search(href)

    url = 'https://public.wmo.int/en/search?search_api_views_fulltext=' + update.message.text

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    quotes = soup.find_all(href=news)
    i = 0
    for link in quotes:
        if link.has_attr('href') and i<=7:
            if i > 2:
                update.message.reply_text('https://public.wmo.int'+link['href'])
            i=i+1

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("5634902583:AAEiLRwgWgMiWEicbXFQaiEsqH3jRu1z3A0", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))
    dp.add_handler(CommandHandler("getlightinf", getlightinf))
    dp.add_handler(CommandHandler("compresstxt", compresstxt))
    dp.add_handler(CommandHandler("findartical", findartical))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    main()
