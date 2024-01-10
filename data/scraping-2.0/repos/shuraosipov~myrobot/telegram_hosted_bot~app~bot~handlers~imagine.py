from collections import deque

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from handlers.utils import send_thinking_message_async
from extentions.chat_gpt import get_image_response

async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Call the OpenAI API Image endpoint to generate an image from a given prompt."""

    # Get the user message
    message = update.message

    # Send the "thinking" message
    thinking_message = await send_thinking_message_async(message)

    # Get image url from openai
    response = await get_image_response(message.text)

    text = f"<a href=\"{response}\">Open image in Browser</a>"

    # Change the "thinking" message with the chatbot's response
    await thinking_message.edit_text(text=text, parse_mode=ParseMode.HTML)