from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackContext,
    MessageHandler,
    filters,
)

from openai_helper import OpenAIHelper


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        self.default_prompt = "What Do you see on this image"
        self.disallowed_message = "Sorry, you are not allowed to use this bot. You can check out the source code at https://github.com/TQMatvey/gpt4-vision-telegram"

    async def is_allowed(
        self, config, update: Update, context: CallbackContext
    ) -> bool:
        if self.config["allowed_user_ids"] == "*":
            return True

        user_id = update.message.from_user.id
        allowed_user_ids = self.config["allowed_user_ids"].split(",")
        if str(user_id) in allowed_user_ids:
            return True

        return False

    async def send_disallowed_message(self, update: Update):
        """
        Sends the disallowed message to the user.
        """
        await update.effective_message.reply_text(
            text=self.disallowed_message, disable_web_page_preview=True
        )

    async def get_image(self, update: Update, context: CallbackContext) -> None:
        # Check if the user is allowed to use this feature
        if not await self.is_allowed(self.config, update, context):
            # If not, send a message and exit the function
            await self.send_disallowed_message(update)
            return

        # Retrieve the highest quality version of the photo sent by the user
        photo_file = await update.message.photo[-1].get_file()

        # Check if there's a caption provided with the photo
        custom_prompt = update.message.caption

        # If a caption is provided, use it as a prompt for image processing
        if custom_prompt:
            await update.message.reply_text(
                self.openai.process_image(photo_file.file_path, custom_prompt)
            )
        else:
            # Otherwise, use a default prompt defined elsewhere in your script
            await update.message.reply_text(
                self.openai.process_image(photo_file.file_path, self.default_prompt)
            )

    async def start(self, update: Update, context: CallbackContext) -> None:
        await update.message.reply_text("""
📷 Please send me a photo with a caption.
The Caption will serve as a prompt for Vision GPT\n
Default prompt is: \"What Do you see on this image\"
""")

    def run(self):
        application = (
            ApplicationBuilder()
            .token(self.config["token"])
            .concurrent_updates(True)
            .build()
        )

        application.add_handler(CommandHandler("start", self.start))
        application.add_handler(MessageHandler(filters.PHOTO, self.get_image))

        application.run_polling()
