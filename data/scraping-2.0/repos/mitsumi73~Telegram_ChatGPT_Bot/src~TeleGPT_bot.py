import os
import logging
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, executor, types
import openai

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)

OPENAI_ORGANIZATION = os.getenv("OPENAI_ORGANIZATION")
openai.organization = OPENAI_ORGANIZATION

# Set up OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# Bot token can be obtained via https://t.me/BotFather
TOKEN = os.getenv("TOKEN")

# Model used in chatGPT
MODEL_NAME = "gpt-3.5-turbo"

# Initialize bot and dispatcher
bot = Bot(token=TOKEN)
dispatcher = Dispatcher(bot)

# Reference object to store the previous response
class Reference:
    def __init__(self):
        self.response = ""

reference = Reference()

def clear_past():
    """A function to clear the previous conversation and context."""
    reference.response = ""

@dispatcher.message_handler(commands=['start'])
async def welcome(message: types.Message):
    """A handler to welcome the user and clear past conversation and context."""
    clear_past()
    await message.reply("Hello! I'm a chatGPT bot!")

@dispatcher.message_handler(commands=['clear'])
async def clear(message: types.Message):
    """A handler to clear the previous conversation and context."""
    clear_past()
    await message.reply("Cleared the past context and chat!")

@dispatcher.message_handler(commands=['help'])
async def helper(message: types.Message):
    """A handler to display the help menu."""
    help_command = """
Hi there, I'm a chatGPT bot! 
Please use these commands:
/start - to start the conversation
/clear - to clear the past conversation and context
/help - to get this help menu
I hope this helps.
"""
    await message.reply(help_command)

@dispatcher.message_handler()
async def chatgpt(message: types.Message):
    try:
        print(f">>> USER: \n\t{message.text}")
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "assistant", "content": reference.response},
                {"role": "user", "content": message.text}
            ]
        )
        reference.response = response['choices'][0]['message']['content']
        print(f">>> chatGPT: {reference.response}")
        await bot.send_message(chat_id=message.chat.id, text=f"{reference.response}")
    except openai.error.OpenAIError as e:
        # Log the error and notify the user
        logging.error(f"OpenAI API error: {e}")
        await message.reply("Sorry, I'm currently unable to respond due to API limitations. Please try again later.")

if __name__ == '__main__':
    print("Starting...")
    executor.start_polling(dispatcher, skip_updates=True)
    print("Stopped")
