import openai
import uuid
import os
import logging
import json
import re

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    MessageHandler,
    filters,
    CommandHandler,
)

# Import necessary modules
import os
import logging
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from constants import PROMPT

# Load environment variables from .env file
load_dotenv()

# Get environment variables
bot_token = os.getenv('BOT_TOKEN')  # Bot token from Telegram API
api_key = os.getenv('API_KEY')  # API key for OpenAI
chat_id = os.getenv('CHAT_ID')  # ID of the chat where the bot will send messages

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

telegram_token = os.environ["TELEGRAM_TOKEN"]
openai.api_key = os.environ["OPENAI_TOKEN"]
os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_TOKEN"]

llm = ChatOpenAI()
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
Memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=1000)
Memory.chat_memory.add_user_message("Could you please remember that I'm keeping my keys in the bedroom?")
Memory.chat_memory.add_ai_message("Noted. Do you happen to have a image showing the exact location?")
Memory.chat_memory.add_user_message("Yes it is in src/fil21.png")

conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=Memory
)

messages_list = []

def is_json(response):
    try:
        json_object = json.loads(response)
        return True
    except ValueError:
        return False
    
def get_json_from_text(text_with_json):
    try:
        # Use regular expression to find the JSON substring within the text
        json_match = re.search(r'{[^}]*}', text_with_json)
        if json_match:
            json_str = json_match.group()

            # Parse the JSON substring into a Python dictionary
            data_dict = json.loads(json_str)
            return data_dict
        else:
            print("No JSON found in the text.")
            return None
    except Exception as e:
        print(f"Error occurred while parsing JSON: {e}")
        return None

def append_history(content, role):
    messages_list.append({"role": role, "content": content})
    return messages_list


def clear_history():
    messages_list.clear()
    return messages_list

async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the photo in ../images"""
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    file_extension = "jpg"
    random_photo_name = f"../images/{uuid.uuid4()}.{file_extension}"
    await photo_file.download_to_drive(random_photo_name)
    logger.info("Photo of %s: %s", user.first_name, random_photo_name)
    conversation({"question": f"Yes it is in {random_photo_name}"})
    await update.message.reply_text(
        "Noted, you're all set."
    )

    return ""


async def process_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    thinking = await context.bot.send_message(
        chat_id=update.effective_chat.id, text="🤔"
    )
    append_history(update.message.text, "user")

    # response = generate_gpt_response()
    response = conversation({"question": update.message.text})
    logger.info(f"response debug {response}")

    append_history(response, "assistant")
    await context.bot.deleteMessage(
        message_id=thinking.message_id, chat_id=update.message.chat_id
    )
    if is_json(str(response['text'])) or is_json(str(response)):
        response = get_json_from_text(str(response['text']))
        if response:
            # response = response['text']
            if response.get('text') is not None:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=str(response['text']))
            if response.get('image_path') not in [None, "None", "null"]:
                # context.bot.send_message(chat_id=update.effective_chat.id, text=str(response['image_path']))
                with open(response.get('image_path'), "rb") as photo:
                    await update.message.reply_photo(photo, caption="Here is your image!")
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="\nImage path not found.")

    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text=str(response['text']))

def generate_gpt_response():
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_list)
    return completion.choices[0].message["content"]

async def reset_history(update, context):
    clear_history()
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Messages history cleaned"
    )
    return messages_list


if __name__ == "__main__":
    application = ApplicationBuilder().token(telegram_token).build()
    text_handler = MessageHandler(
        filters.TEXT & (~filters.COMMAND), process_text_message
    )
    application.add_handler(text_handler)

    application.add_handler(CommandHandler("reset", reset_history))

    photo_handler = MessageHandler(filters.PHOTO, photo)
    application.add_handler(photo_handler)

    # audio_handler = MessageHandler(filters.VOICE, process_audio_message)
    # application.add_handler(audio_handler)

    application.run_polling()