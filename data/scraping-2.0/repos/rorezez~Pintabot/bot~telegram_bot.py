from __future__ import annotations
from openAI_helper import OpenAIHelper
from telegram import BotCommand
from telegram import Update, constants
from telegram.ext import ContextTypes, ApplicationBuilder, CommandHandler, MessageHandler, filters, InlineQueryHandler, Application
from utils import message_text, split_into_chunks_nostream , wrap_with_indicator, error_handler
import logging




# Path: bot/helpers/telegram_helper.py
class TelegramHelper:
    "ini adalah class untuk telegramnya"

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        self.assistant_id = self.openai.get_or_create_assistant(name="Pintabot")
        self.commands = [
            BotCommand(command='help', description="Menampilkan pesan bantuan")
        ]
        self.disallowed_message = "Maaf anda belum terdaftar sebagai pengguna bot ini. Silahkan hubungi @{} untuk mendapatkan akses.".format(
            self.config['admin_usernames'])
        self.last_message = {}

    async def help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Shows the help menu.
        """
        commands = self.commands
        commands_description = [f'/{command.command} - {command.description}' for command in commands]
        help_message = f"Selamat datang di {self.config['bot_name']}!\n\n"

        # Menambahkan daftar perintah ke help_message
        help_message += "Berikut adalah command yang bisa di gunakan:\n"
        help_message += '\n'.join(commands_description)
        help_message += "\n\nUntuk mulai menggunakan {}, silahkan ajukan pertanyaan melalui pesan text atau menggunakan pesan suara.".format(self.config['bot_name'])
        
        await update.message.reply_text(help_message, disable_web_page_preview=True)
    
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handles the user's prompt and sends the response.
        """
        logging.info(
            f'Pesan baru dari user {update.message.from_user.name} (id: {update.message.from_user.id})')
        chat_id = update.effective_chat.id
        query = f"{message_text(update.message)}"
        self.last_message[chat_id] = query
        try:
            async def _reply():
                # Get response from OpenAI
                messages = await self.openai.get_message_from_assistant(chat_id=chat_id, prompt=query)

                # Filter out only the assistant's messages
                assistant_messages = [msg for msg in reversed(messages) if msg.role == "assistant"]

                # Check if there are any assistant messages
                if assistant_messages:
                    # Get the last message from the assistant
                    last_response = assistant_messages[-1].content[0].text  # Sesuaikan akses atribut
                    text = last_response.value  # Sesuaikan akses atribut
                    # Process the last response (e.g., split into chunks if necessary)
                    # Assuming last_response is a string; modify as needed based on actual data structure
                    chunks = split_into_chunks_nostream(text)

                    for chunk in chunks:
                        try:
                            await update.effective_message.reply_text(
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except Exception:
                            try:
                                await update.effective_message.reply_text(text=chunk)
                            except Exception as exception:
                                        raise exception
            logging.info("Starting wrap_with_indicator function")
            await wrap_with_indicator(update, context, _reply, constants.ChatAction.TYPING)
            logging.info("Finished wrap_with_indicator function")

        except Exception as e:
            logging.exception(e)
            await update.effective_message.reply_text(
                text="Maaf, terjadi kesalahan saat memproses permintaan anda. Silahkan coba lagi nanti.",
                parse_mode=constants.ParseMode.MARKDOWN
            )
    
    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            """
            Handle the inline query. This is run when you type: @botusername <query>
            """
            query = update.inline_query.query
            if len(query) < 3:
                return
            if not await self.check_allowed_and_within_budget(update, context, is_inline=True):
                return

            callback_data_suffix = "gpt:"
            result_id = str(uuid4())
            self.inline_queries_cache[result_id] = query
            callback_data = f'{callback_data_suffix}{result_id}'

            await self.send_inline_query_result(update, result_id, message_content=query, callback_data=callback_data)

    async def post_init(self, application: Application) -> None:
            """
            Post initialization hook for the bot.
            """
            await application.bot.set_my_commands(self.commands)

    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CommandHandler('start', self.help))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP, constants.ChatType.PRIVATE
        ]))

        application.add_error_handler(error_handler)

        application.run_polling()