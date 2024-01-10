import json
import logging
import os
import openai
from chat_utils import ask

# Load config values
with open('config.json') as config_file:
    config_details = json.load(config_file)

chatgpt_model_name = config_details['CHATGPT_MODEL']
openai.api_type = "azure"
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = config_details['OPENAI_API_BASE']
openai.api_version = config_details['OPENAI_API_VERSION']
bearer_token_db = os.environ['BEARER_TOKEN']

if __name__ == "__main__":
    while True:
        user_query = input("Enter your question: ")
        print("query...")
        # openai.api_key = open_ai_gpt_api_key
        # openai.api_base = open_ai_api_base
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s %(levelname)s %(message)s")
        print(ask(user_query, bearer_token_db))