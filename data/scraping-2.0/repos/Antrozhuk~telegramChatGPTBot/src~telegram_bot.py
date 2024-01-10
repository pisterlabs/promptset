import telegram.constants as constants
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from src.openai_helper import OpenAIHelper
from src.logger import Logger


class ChatGPT3TelegramBot:
    """
    Class representing a Chat-GPT3 Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Ініціалізує бот конфігурацією та GPT-3 налаштуваннями.
        :param config: Словник з конфігурацією бота
        :param openai: OpenAIHelper обʼєкт
        :param disallowed_message: Повідомлення про відсутність доступу
        """
        self.config = config
        self.openai = openai
        self.logger = Logger('telegram_bot').get_logger()
        self.disallowed_message = "Вибачте, але вам не дозволено користуватись цим ботом."

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Показує початкове повідомлення.
        """
        if await self.disallowed(update, context):
            return

        await update.message.reply_text("Привіт! Я бот, який відповідає на ваші повідомлення за допомогою ChatGPT-3.\n"
                                        "Якщо ви хочете дізнатись більше про мене, введіть /help\n\n",
                                        disable_web_page_preview=True)

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Показує допоміжне повідомлення.
        """
        if await self.disallowed(update, context):
            return

        await update.message.reply_text("[Будь яке повідомлення] - Відправляє ваше повідомлення до AI\n"
                                        "/help - Меню помічника\n"
                                        "/random_answer - Генерує рандомну відповідь\n"
                                        "/random_post - Генерує рандомний пост\n"
                                        "/reset - Оновлює бесіду\n\n",
                                        disable_web_page_preview=True)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Оновлює бесіду.
        """
        if await self.disallowed(update, context):
            return

        self.logger.info(f'Resetting the conversation for {update.message.from_user}...')

        chat_id = update.effective_chat.id
        self.openai.reset_chat_history(chat_id=chat_id)
        await context.bot.send_message(chat_id=chat_id, text='Готово!')

    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        if await self.disallowed(update, context):
            return

        self.logger.info(f'New message "{update.message.text}" received from {update.message.from_user}')
        chat_id = update.effective_chat.id
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

        response = self.openai.get_chat_response(chat_id=chat_id, query=update.message.text)
        await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=update.message.id,
            parse_mode=constants.ParseMode.MARKDOWN,
            text=response
        )

    async def random_answer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Відправляє рандомну відповідь.
        """
        if await self.disallowed(update, context):
            return

        self.logger.info(f'random_answer command received from {update.message.from_user}')
        chat_id = update.effective_chat.id
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

        response = self.openai.get_chat_response(chat_id=chat_id, query='напиши рандомну відповідь')
        await context.bot.send_message(
            chat_id=chat_id,
            reply_to_message_id=update.message.id,
            parse_mode=constants.ParseMode.MARKDOWN,
            text=response
        )

    async def random_post(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Відправляє рандомний пост.
        """
        if await self.disallowed(update, context):
            return

        self.logger.info(f'random_post command received from {update.message.from_user}')
        chat_id = update.effective_chat.id
        await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)

        response = self.openai.get_chat_response(chat_id=chat_id, query='напиши рандомний пост українською')
        await context.bot.send_message(
            chat_id=chat_id,
            parse_mode=constants.ParseMode.MARKDOWN,
            text=response
        )

    async def disallowed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Відправляє повідомлення про відсутність доступів до користувача.
        """
        if not await self.is_allowed(update):
            self.logger.warning(f'User {update.message.from_user} is not allowed to use the bot')
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=self.disallowed_message,
                disable_web_page_preview=True
            )
            return True
        return False

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Відловлює всі помилки.
        """
        self.logger.debug(f'Exception while handling an update: {context.error}')

    async def is_allowed(self, update: Update) -> bool:
        """
        Перевіряє чи дозволено юзеру користуватись даним ботом.
        """
        if self.config['allowed_user_ids'] == '*':
            return True

        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        if str(update.message.from_user.id) in allowed_user_ids:
            return True

        return False

    def run(self):
        """
        Запускає бот доки користувач не натисне Ctrl+C
        """
        application = ApplicationBuilder().token(self.config['token']).build()

        application.add_handler(CommandHandler('start', self.start))

        application.add_handler(CommandHandler('help', self.help))

        application.add_handler(CommandHandler('reset', self.reset))

        application.add_handler(CommandHandler('random_answer', self.random_answer))

        application.add_handler(CommandHandler('random_post', self.random_post))

        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))

        application.add_error_handler(self.error_handler)

        application.run_polling()
