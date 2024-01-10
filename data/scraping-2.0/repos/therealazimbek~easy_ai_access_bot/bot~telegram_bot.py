import logging

from pydub import AudioSegment
from telegram import BotCommand, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    CallbackContext,
    Application,
    MessageHandler,
    CommandHandler,
    filters,
    ContextTypes, CallbackQueryHandler,
)

from clients.gemini_client import GeminiClient
from clients.openai_client import OpenAIClient
from clients.vision_client import VisionClient
from repositories.user_repository import UserRepository
from utils.delete_file import delete_file_if_exists
from utils.token_counter import validate_user_input

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(
            self, openai_client: OpenAIClient, vision_client: VisionClient, gemini_client: GeminiClient, config: dict
    ):
        self.openai_client = openai_client
        self.vision_client = vision_client
        self.gemini_client = gemini_client
        self.repository = UserRepository()
        self.config = config
        self.commands = [
            BotCommand(command="help", description="Show help message"),
            BotCommand(command="start", description="Show welcome message"),
            BotCommand(command="stats", description="Show user statistics"),
            BotCommand(command="state", description="Show currently chosen service"),
            BotCommand(command="menu", description="Show services menu"),
        ]
        self.keyboard = [[InlineKeyboardButton("ChatGPT4-Turbo", callback_data='gpt'),
                          InlineKeyboardButton("Text to Speech", callback_data='tts')],
                         [InlineKeyboardButton("Image to Text", callback_data='itt'),
                          InlineKeyboardButton("Image generation", callback_data='dalle')],
                         [InlineKeyboardButton("Audio transcribing", callback_data='att'),
                          InlineKeyboardButton("Google Gemini", callback_data='gemini')]]

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user = update.effective_user

        await self.add_user_to_db(user)

        await update.message.reply_text(
            f"Hello {user.name}! Welcome to EasyAIAccess Bot, where you can access AI services easily and quickly!"
        )

        await self.show_menu(update, context)

    async def add_user_to_db(self, user):
        if not self.repository.user_exists(user.id):
            self.repository.insert_user(
                user.id, user.username, user.first_name, user.last_name
            )

    async def help_command(
            self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        await self.add_user_to_db(update.effective_user)
        message = ("You confused and do not know how to use this bot? Please, use /menu command to get a list of "
                   "available services and choose one of them. Then start entering inputs or sending documents, "
                   "depending on chosen service. If you are confused in which service you are, just use /state "
                   "command to get your current chosen service. Please, keep in mind that responses can take a while "
                   "if they are big, be patient. You can also contact me through Telegram!\nThanks for using this "
                   "bot! EasyAIAccess Bot by @therealazimbek")
        await update.message.reply_text(message)

    async def generate_gpt_response(
            self, update: Update, context: CallbackContext
    ) -> None:
        await self.add_user_to_db(update.effective_user)

        user_input = update.message.text.strip()

        if validate_user_input(user_input):
            await update.message.reply_text(
                "Please wait, your request is processing, for large responses it can take a while!"
            )
            logger.info(f"User {update.effective_user.id}: input sent to gpt model...")
            generated_text = await self.openai_client.generate_response(user_input)
            await update.message.reply_text(generated_text)
            logger.info(f"User {update.effective_user.id}: response sent back...")
            self.repository.update_request_count(update.effective_user.id, "gpt")
        else:
            await update.message.reply_text(
                "Too many characters. Please try again with less characters."
            )

    async def generate_gemini_response(
            self, update: Update, context: CallbackContext
    ) -> None:
        await self.add_user_to_db(update.effective_user)

        user_input = update.message.text.strip()

        if validate_user_input(user_input):
            await update.message.reply_text(
                "Please wait, your request is processing, for large responses it can take a while!"
            )
            logger.info(f"User {update.effective_user.id}: input sent to gemini model...")
            generated_text = await self.gemini_client.generate_response(user_input)
            await update.message.reply_text(generated_text)
            logger.info(f"User {update.effective_user.id}: response sent back...")
            self.repository.update_request_count(update.effective_user.id, "gemini")
        else:
            await update.message.reply_text(
                "Too many characters. Please try again with less characters."
            )

    async def stats_command(self, update: Update, context: CallbackContext) -> None:
        await self.add_user_to_db(update.effective_user)

        result = self.repository.get_service_counts(update.effective_user.id)
        result_string = "Your total request so far:\n"
        for service, count in result.items():
            service_name = "-".join(word.capitalize() for word in service.split("-"))
            result_string += f"{service_name}: {count}\n"

        await update.message.reply_text(result_string.strip())

    async def unrecognized_command(
            self, update: Update, context: CallbackContext
    ) -> None:
        await self.add_user_to_db(update.effective_user)
        await update.message.reply_text(
            "Sorry, I don't understand that command. See /help"
        )

    async def image_command(self, update: Update, context: CallbackContext) -> None:
        await self.add_user_to_db(update.effective_user)

        user_input = update.message.text.replace("/image", "").strip()

        if validate_user_input(user_input):
            await update.message.reply_text(
                "Please wait, your request is processing, for large responses and images it can take a while!"
            )
            logger.info(f"User {update.effective_user.id}: input sent to dalle model...")
            response = await self.openai_client.generate_image(user_input)
            await update.message.reply_photo(response)
            logger.info(f"User {update.effective_user.id}: response sent back...")
            self.repository.update_request_count(
                update.effective_user.id, "image-generation"
            )
        else:
            await update.message.reply_text(
                "Please provide valid input. Example: /image cute cat"
            )

    async def tts_command(self, update: Update, context: CallbackContext) -> None:
        await self.add_user_to_db(update.effective_user)

        user_input = update.message.text.replace("/tts", "").strip()

        if validate_user_input(user_input):
            await update.message.reply_text(
                "Please wait, your request is processing, for large responses it can take a while!"
            )
            logger.info(
                f"User {update.effective_user.id}: input sent to text-to-speech model..."
            )
            response = await self.openai_client.generate_speech(user_input)
            await update.message.reply_voice(response)
            logger.info(f"User {update.effective_user.id}: response sent back...")
            self.repository.update_request_count(
                update.effective_user.id, "text-to-speech"
            )
            if response.exists():
                response.unlink()
        else:
            await update.message.reply_text(
                "Please provide valid input. Example: /tts Hello from ai speech"
            )

    async def image_to_text(self, update: Update, context: CallbackContext) -> None:
        await self.add_user_to_db(update.effective_user)

        if update.message.photo:
            input_image_id = update.message.photo[-1].file_id
        elif update.message.document and update.message.document.mime_type.startswith(
                "image"
        ):
            input_image_id = update.message.document.file_id
        else:
            await update.message.reply_text("Please send a valid photo or image file.")
            return

        image_name = f"{input_image_id}.jpeg"
        input_image = await context.bot.get_file(input_image_id)
        await input_image.download_to_drive(image_name)

        with open(image_name, "rb") as image_file:
            content = image_file.read()

        await update.message.reply_text(
            "Please wait, your request is processing, for large responses it can take a while!"
        )
        logger.info(f"User {update.effective_user.id}: input sent to google vision...")
        response = await self.vision_client.image_to_text_client(content)
        await update.message.reply_text(response)
        logger.info(f"User {update.effective_user.id}: response sent back...")
        self.repository.update_request_count(update.effective_user.id, "image-to-text")

        delete_file_if_exists(image_name)

    async def transcribe_command(
            self, update: Update, context: CallbackContext
    ) -> None:
        await self.add_user_to_db(update.effective_user)

        filename = update.message.effective_attachment.file_unique_id
        filename_mp3 = f"{filename}.mp3"
        media_file = await context.bot.get_file(
            update.message.effective_attachment.file_id
        )
        await media_file.download_to_drive(filename)
        audio_track = AudioSegment.from_file(filename)
        audio_track.export(filename_mp3, format="mp3")

        audio_file = open(filename_mp3, "rb")
        await update.message.reply_text(
            "Please wait, your request is processing, for large responses it can take a while!"
        )
        logger.info(
            f"User {update.effective_user.id}: input sent to transcribe model..."
        )
        generated_text = await self.openai_client.transcribe_audio(audio_file)
        await update.message.reply_text("Transcribed text: " + generated_text)
        logger.info(f"User {update.effective_user.id}: response sent back...")
        self.repository.update_request_count(update.effective_user.id, "audio-to-text")
        audio_file.close()

        delete_file_if_exists(filename_mp3)
        delete_file_if_exists(filename)

    async def show_menu(self, update: Update, context: CallbackContext) -> None:
        reply_markup = InlineKeyboardMarkup(self.keyboard)

        await update.message.reply_text('Please choose a service:', reply_markup=reply_markup)

    async def update_handler(self, update: Update, context: CallbackContext) -> None:
        state = self.repository.get_user_state(update.effective_user.id)

        if state == "gpt":
            if update.message.text:
                await self.generate_gpt_response(update, context)
            else:
                await update.message.reply_text("For this service, please, send only text messages!")
        elif state == "tts":
            if update.message.text:
                await self.tts_command(update, context)
            else:
                await update.message.reply_text("For this service, please, send only text messages!")
        elif state == "itt":
            if update.message.photo or (update.message.document and update.message.document.mime_type.startswith(
                    "image"
            )):
                await self.image_to_text(update, context)
            else:
                await update.message.reply_text("For this service, please, send only images or documents with image "
                                                "types!")
        elif state == "dalle":
            if update.message.text:
                await self.image_command(update, context)
            else:
                await update.message.reply_text("For this service, please, send only text messages!")
        elif state == "att":
            if (update.message.audio or update.message.voice or
                    (update.message.document and update.message.document.mime_type.startswith(
                        "audio"
                    ))):
                await self.transcribe_command(update, context)
            else:
                await update.message.reply_text("For this service, please, send only voice messages or audios "
                                                "or documents with audio types!")
        elif state == "gemini":
            if update.message.text:
                await self.generate_gemini_response(update, context)
            else:
                await update.message.reply_text("For this service, please, send only text messages!")
        else:
            await update.message.reply_text("Please choose service first using /menu command!")

    async def keyboard_handler(self, update: Update, context: CallbackContext) -> None:
        query = update.callback_query
        user_id = query.from_user.id
        choice = query.data

        if choice == "gpt":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose ChatGPT4-Turbo, latest OpenAI Language Model. Start typing "
                                           "requests!")
            self.repository.set_user_state(user_id, 'gpt')
        elif choice == "tts":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose Text to Speech from OpenAI. Start typing requests!")
            self.repository.set_user_state(user_id, 'tts')
        elif choice == "itt":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose Image to Text from Google Vision. Start sending images!")
            self.repository.set_user_state(user_id, 'itt')
        elif choice == "dalle":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose Image generation from OpenAI DALLE-3. Start typing "
                                           "requests!")
            self.repository.set_user_state(user_id, 'dalle')
        elif choice == "att":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose Audio transcribing from OpenAI. Start sending "
                                           "voice messages or audio files!")
            self.repository.set_user_state(user_id, 'att')
        elif choice == "gemini":
            await context.bot.send_message(update.effective_chat.id,
                                           "You chose Gemini Language model, just like GPT but from Google. "
                                           "Start typing requests!")
            self.repository.set_user_state(user_id, 'gemini')

    async def show_state_command(self, update: Update, context: CallbackContext) -> None:
        state = self.repository.get_user_state(update.effective_user.id)
        modified_state = ''

        if state == "gpt":
            modified_state = 'ChatGPT4-Turbo'
        elif state == "tts":
            modified_state = 'Text to Speech'
        elif state == "itt":
            modified_state = 'Image to Text'
        elif state == "dalle":
            modified_state = 'Image generation'
        elif state == "att":
            modified_state = 'Audio transcribing'
        elif state == "gemini":
            modified_state = 'Google Gemini'

        await update.message.reply_text("Your current state: " +
                                        modified_state)

    async def post_init(self, application: Application) -> None:
        await application.bot.set_my_commands(self.commands)

    def run(self):
        application = (
            Application.builder()
            .token(self.config["token"])
            .post_init(self.post_init)
            .build()
        )

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(CommandHandler("stats", self.stats_command))
        application.add_handler(CommandHandler("help", self.help_command))
        application.add_handler(CommandHandler("state", self.show_state_command))
        application.add_handler(CommandHandler("menu", self.show_menu))
        application.add_handler(
            MessageHandler(filters.COMMAND, self.unrecognized_command)
        )
        application.add_handler(MessageHandler((filters.TEXT | filters.AUDIO |
                                                filters.VOICE | filters.Document.AUDIO |
                                                filters.PHOTO | filters.ATTACHMENT) & ~filters.COMMAND,
                                               self.update_handler))
        application.add_handler(CallbackQueryHandler(self.keyboard_handler))

        application.run_polling(allowed_updates=Update.ALL_TYPES)
