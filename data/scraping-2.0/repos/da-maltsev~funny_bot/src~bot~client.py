from dataclasses import dataclass
import logging
from typing import Callable, Optional

from telegram import Update
from telegram.ext import ApplicationBuilder
from telegram.ext import CommandHandler
from telegram.ext import ContextTypes
from telegram.ext import filters
from telegram.ext import MessageHandler

from bot.filters import ask_question_filter
from bot.filters import make_picture_filter
from bot.help_text import help_text
from bot.services import AskQuestionService
from bot.services import PictureMaker
from clients.open_ai.client import OpenaiClient

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)


@dataclass
class TelegramBot:
    token: str
    open_ai_client: OpenaiClient

    def __post_init__(self):
        self.application = ApplicationBuilder().token(self.token).build()

    def __call__(self):
        self.add_handlers()
        self.application.run_polling()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logging.warning("Someone started")
        user_name = f"{self.user_name_from_update(update)}!\n" if self.user_name_from_update(update) else None
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"{user_name}{help_text.get('start', 'Hello there!')}")  # type: ignore

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,  # type: ignore
            text=help_text.get("help", "Just figure it out bro"),
        )

    async def make_picture(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logging.info(update.message and update.message.text or "")  # type: ignore
        user_name = self.user_name_from_update(update)
        await PictureMaker(self.open_ai_client, context)(update.effective_chat.id, update.message.text, user_name)  # type: ignore

    async def ask_question(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        logging.info(update.message and update.message.text or "")  # type: ignore
        user_name = self.user_name_from_update(update)
        await AskQuestionService(self.open_ai_client, context)(update.effective_chat.id, update.message.text, user_name)  # type: ignore

    def add_handlers(self) -> None:
        self.add_commands(
            [
                self.start,
                self.help,
            ]
        )
        self.add_message_handlers()

    def add_commands(self, functions: list[Callable]) -> None:
        for func in functions:
            self.application.add_handler(CommandHandler(func.__name__, func))

    def add_message_handlers(self) -> None:
        self.application.add_handler(MessageHandler(make_picture_filter & (~filters.COMMAND), self.make_picture))
        self.application.add_handler(MessageHandler(ask_question_filter & (~filters.COMMAND), self.ask_question))

    @classmethod
    def user_name_from_update(cls, update: Update) -> Optional[str]:
        message = update.message
        if not message:
            return None
        return message.from_user and message.from_user.first_name or None
