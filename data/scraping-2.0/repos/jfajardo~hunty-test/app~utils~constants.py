import os

import openai

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
APP_URL = os.getenv('APP_URL')
TELEGRAM_URL_API = os.getenv('TELEGRAM_URL_API')
openai.api_key = os.getenv('OPENAI_API_KEY')
