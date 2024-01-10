import os

import openai
from dotenv import load_dotenv

# Find .env file
load_dotenv()

# OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')

# Telegram bot key
tgkey = os.getenv('TELEGRAM_KEY')

# Defaults
main_user_id = int(os.getenv('MAIN_USER_ID'))
available_in_chat = os.getenv('AVAILABLE_IN_CHATS').split(',')
botname = os.getenv('BOT_NAME')

# Lots of console output
debug = False

# Wait for user to answer on greeting message
minutes_for_user_thinking = 10
