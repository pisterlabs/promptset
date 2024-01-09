import logging

import telegram.constants as constants
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from openai_helper import OpenAIHelper


class ChatGPT3TelegramBot:
    """
    Class representing a Chat-GPT3 Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializing bot config and GPT-3 settings.
        :param config: Bot config
        :param openai: OpenAIHelper object
        :param disallowed_message: Message about permission issue
        """
        self.config = config
        self.openai = openai
        self.disallowed_message = "Sorry, but you do not have permissions to use bot."

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Show user guide.
        """
        await update.message.reply_text("/reset - update conversation\n"
                                        "[Any message] - Send you request to AI\n"
                                        "/help - Helper menu\n\n",
                                        disable_web_page_preview=True)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Update conversation.
        """
        if not await self.is_allowed(update):
            logging.warning(f'User {update.message.from_user.name} is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name}...')

        chat_id = update.effective_chat.id
        self.openai.reset_chat_history(chat_id=chat_id)
        await context.bot.send_message(chat_id=chat_id, text='Done!')

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        chat_id = update.effective_chat.id
        if not await self.is_allowed(update):
            logging.warning(f'User {update.message.from_user.name} is not allowed to use this bot')

            await self.send_disallowed_message(update, context)
            return

        logging.info(f'New message received from user {update.message.from_user.id}')
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

        response = self.openai.get_chat_response(chat_id=chat_id, query=update.message.text)
        logging.info("Response from ChatGPT received")

        await context.bot.send_message(
            text=response,
            chat_id=chat_id,
            reply_to_message_id=update.message.message_id,
            parse_mode=constants.ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    async def send_disallowed_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        await context.bot.send_message(
            text=self.disallowed_message,
            chat_id=chat_id,
            disable_web_page_preview=True,
        )
    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Catch all errors
        """
        logging.debug(f'Exception while handling an update: {context.error}')

    async def is_allowed(self, update: Update) -> bool:
        """
        Check user permissions for bot
        """
        if self.config["allowed_user_ids"] == "*":
            return True
        allowed_user_ids = self.config["allowed_user_ids"].split(",")

        if  str(update.message.from_user.id) in allowed_user_ids:
            return True

        return False

    def run(self):
        """
        Run bot until user close it with Ctrl+C
        """
        application = ApplicationBuilder().token(self.config['token']).build()

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('start', self.help))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))

        application.add_error_handler(self.error_handler)

        application.run_polling()
