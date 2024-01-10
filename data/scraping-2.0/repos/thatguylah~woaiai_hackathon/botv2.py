"""
Image-generating Bot that assists with image design ideas and image generation based on its own text-to-image prompt

Usage:
Sequence of automated questions to answer and then generating an image based on suggested image design.
Press Ctrl-C on the command line or send a signal to the process to stop the bot.
"""
import openai
import logging
from dotenv import dotenv_values
import argparse
import os
from api.conversation import (
    start_command,
    cancel_command,
    get_image_purpose,
    select_theme,
    select_image_design,
    get_image_prompt,
    generate_image,
)
from api.utils import run_in_threadpool_decorator
from api.outpainting import outpainting_handler

from telegram import __version__ as TG_VER
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    ConversationHandler,
    PicklePersistence,
)

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This example is not compatible with your current PTB version {TG_VER}. To view the "
        f"{TG_VER} version of this example, "
        f"visit https://docs.python-telegram-bot.org/en/v{TG_VER}/examples.html"
    )

# get config
config = dotenv_values(".env")
# get API tokens
HF_TOKEN = config["HF_API_KEY"]
openai.api_key = config["OPENAI_API_KEY"]
# TELEBOT_TOKEN = config['TELEBOT_TOKEN']
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(processName)s - %(threadName)s - [%(thread)d] - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
huggingFaceLogger = logging.getLogger("huggingface_hub").setLevel(logging.INFO)
messageHandlerLogger = logging.getLogger("telegram.bot").setLevel(logging.INFO)
applicationLogger = logging.getLogger("telegram.ext").setLevel(logging.INFO)
openAILogger = logging.getLogger("openai").setLevel(logging.INFO)


# assign variable name for each integer in sequence for easy tracking of conversation
(
    IMAGE_TYPE,
    IMAGE_PURPOSE,
    SELECTED_THEME,
    SELECTED_IMAGE_DESIGN,
    TEXT_TO_IMAGE_PROMPT,
    GENERATED_IMAGE,
) = range(6)


async def pong(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """serves a health check status for debugging

    Args:
        update (Update): _description_
        context (ContextTypes.DEFAULT_TYPE): _description_
    Returns:
        Pong to the user
    """
    await update.message.reply_text("Pong")


# function to start the bot
def main(dev_mode) -> None:
    if dev_mode:
        TELEBOT_TOKEN = config["TELEBOT_DEV_TOKEN"]
    else:
        TELEBOT_TOKEN = config["TELEBOT_TOKEN"]

    # create folders for outputs
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/image_output"):
        os.mkdir("data/image_output")

    # configure chatbot's persistence
    persistence = PicklePersistence(filepath="data/conversation")

    # create the Application pass telebot's token to application
    application = (
        Application.builder()
        .token(TELEBOT_TOKEN)
        .concurrent_updates(True)
        .persistence(persistence)
        .build()
    )

    # Conversation Handler with the states IMAGE_TYPE, IMAGE_PURPOSE, SELECTED_THEME, SELECTED_IMAGE_DESIGN
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start_command)],
        states={
            IMAGE_TYPE: [
                CommandHandler("Yes", start_command),
                CommandHandler("No", cancel_command),
            ],
            IMAGE_PURPOSE: [
                MessageHandler(filters.TEXT, get_image_purpose, block=False)
            ],
            SELECTED_THEME: [MessageHandler(filters.TEXT, select_theme, block=False)],
            SELECTED_IMAGE_DESIGN: [
                MessageHandler(filters.TEXT, select_image_design, block=False)
            ],
            TEXT_TO_IMAGE_PROMPT: [
                MessageHandler(filters.TEXT, get_image_prompt, block=False)
            ],
            GENERATED_IMAGE: [
                MessageHandler(filters.TEXT, generate_image, block=False)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
        allow_reentry=True,  # allow user to enter back any state of the ConversationHandler
        name="ImageGeneratingBot",
        persistent=True,
        block=False,
    )

    ping_handler = CommandHandler("ping", pong, block=False)

    # add conversation handler to application
    application.add_handler(conv_handler)
    application.add_handler(ping_handler)
    application.add_handler(outpainting_handler)
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-DEV", "--dev", action="store_true", help="Run with local Tele API token"
    )
    args = parser.parse_args()

    main(args.dev)
