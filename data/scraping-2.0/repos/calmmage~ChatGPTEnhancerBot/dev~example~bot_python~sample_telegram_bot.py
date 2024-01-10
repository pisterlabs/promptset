# a test of echo telegram bot, that is somehow more advanced that just a stupid echo bot / default telegram stuff

# Features to implement:
# 0) Know who the user is ?
# 1) Graceful response in case of errors
# catch errors and handle them. Telegram has error handlers?
# 2) User type support
# Option 1: admin vs regular user. Admin: gets full error diagnostics. User: gets "sorry was an error" message.
# Option 2: some commands unavailable
# 3)

"""a simple bot that just forwards queries to openai and sends the response"""

import logging
from functools import wraps

from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# Load the secrets from a file
from chatgpt_enhancer_bot.main import send_menu  # , build_menu

secrets = {}
with open("secrets.txt", "r") as f:
    for line in f:
        key, value = line.strip().split(":", 1)
        secrets[key] = value

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and
# context.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    update.message.reply_markdown_v2(
        fr'Hi {user.mention_markdown_v2()}\!',
        reply_markup=ForceReply(selective=True),
    )


class SampleCommandSource:
    commands = {
        '/test': 'sample_command',
        '/help': 'help'
    }

    def test_command(self):
        return "Test command activated"

    def help(self):
        return "Help command activated"

    @staticmethod
    def parse_query(query):
        """format: "/command arg1 arg2 key3=arg3" """
        parts = query.strip().split()
        if parts[0].startswith('/'):
            command = parts[0]
            parts = parts[1:]
        else:
            raise RuntimeError(f"command not included? {query}")
        args = []
        kwargs = {}
        for p in parts:
            if '=' in p:
                k, v = p.split('=')
                kwargs[k] = v
            else:
                args.append(p)
        return command, args, kwargs


# todo: add custom handler for this menu command
#  consider adding this to list_topics command
#  and to swith_topic command, if there's no args.. - here, definitely no harm.

# the question is how to trigger the menu from inside the bot?
# def topics_menu(self, )l.,

# let me think abit
# 1) I can add this to telegram bot. But then it won't know that kwargs are missing. Or can I detect it in command handler?
# 2) Or I can do this from openai api. But then how do I send the message?


def sample_menu_handler(update, context):
    menu = {
        'Option 1': 'I am selecting option1',
        'Option 2': 'I am selecting option2',
        'Option 3': 'I am selecting option3',
    }
    send_menu(update, context, menu, 'Select an option')


# def help_command(update: Update, context: CallbackContext) -> None:
#     """Send a message when the command /help is issued."""
#     update.message.reply_text('Help!')


def telegram_decorator(func, **kwargs):
    @wraps(func)
    def wrapper(update: Update, context: CallbackContext) -> None:
        response = func(update.message.text, **kwargs)
        update.message.reply_text(response)

    return wrapper


def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    reply_text_message = f"We've got your message. It is: {update.message.text}"
    reply_text_message += f"\n And the user is: {update.effective_user.username}"
    update.message.reply_text(reply_text_message)


def main(expensive: bool) -> None:
    """
    Start the bot
    :param expensive: Use 'text-davinci-003' model instead of 'text-ada:001'
    :return:
    """
    # Create the Updater and pass it your bot's token.
    token = secrets["telegram_api_token"]
    updater = Updater(token)

    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    # dispatcher.add_handler(CommandHandler("help", help_command))
    # add commands
    cs = SampleCommandSource()

    def make_command_handler(method_name):
        def command_handler(update: Update, context: CallbackContext) -> None:
            bot = cs
            method = bot.__getattribute__(method_name)

            prompt = update.message.text
            command, qargs, qkwargs = bot.parse_query(prompt)
            result = method(*qargs, **qkwargs)  # todo: parse kwargs from the command
            update.message.reply_text(result)

        return command_handler

    for command, method_name in SampleCommandSource.commands.items():
        # logger.info(command, method_name )
        # func = cs.__getattribute__()
        # method_name = SampleCommandSource.commands[command]
        handler = make_command_handler(method_name)
        dispatcher.add_handler(CommandHandler(command.lstrip('/'), handler))

    model = "text-davinci-003" if expensive else "text-ada:001"
    # on non command i.e message - echo the message on Telegram
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command,
                                          echo))

    dispatcher.add_handler(CommandHandler("sample_menu", sample_menu_handler))

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--expensive", action="store_true",
                        help="use expensive calculation - 'text-davinci-003' model instead of 'text-ada:001' ")
    args = parser.parse_args()

    main(expensive=args.expensive)
