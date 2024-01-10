import os

import openai
from dotenv import load_dotenv

# Find .env file
load_dotenv()

version = "0.2.0"

# OpenAI API key
openai.api_key = os.getenv('OPENAI_KEY')

# Telegram bot key
tgkey = os.getenv('TELEGRAM_KEY')

# Defaults
main_users_id = [int(numeric_string) for numeric_string in os.getenv('MAIN_USERS_ID').split(',')]
chats_and_greetings = dict(map(lambda pair: pair.split(":"), os.getenv('CHATS_GREETINGS').split(';')))
botname = os.getenv('BOT_NAME')

# Lots of console output
debug = False

# Wait for user to answer on greeting message
minutes_for_user_thinking = 10

# How many messages to save for each user
store_last_messages = 20

# How long store messages in cache
message_cache_minutes = 10

# Attempts for making request to OpenAI
total_attempts = 5

# Will be filled after start
bot_id = None
