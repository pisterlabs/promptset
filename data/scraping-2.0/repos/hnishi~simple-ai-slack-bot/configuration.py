import os

import openai
from dotenv import load_dotenv
from slack_sdk import WebClient

if os.path.exists(".env"):
    load_dotenv(".env")
else:
    raise FileNotFoundError("No .env file found. Please create one.")

IS_MESSAGE_SAVE_ENABLED = False  # Save messages to sqlite database

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# See https://platform.openai.com/docs/models/gpt-4
if MODEL_NAME.startswith("gpt-4"):
    MODEL_MAX_TOKEN_LENGTH = 8192
elif MODEL_NAME.startswith("gpt-3.5-turbo"):
    MODEL_MAX_TOKEN_LENGTH = 4097
else:
    supported_models = ["gpt-3.5-turbo", "gpt-4"]
    raise NotImplementedError(
        f"Unknown model {MODEL_NAME} is provided.\n"
        f"Supported models are: {', '.join(supported_models)}"
    )

SYSTEM_PROMPT = f"""
あなたは OpenAI API の {MODEL_NAME} モデルを利用した Slack Bot です。
これから行われる一連の会話は、Slack スレッド上で行われるものです。
過去のスレッドメッセージも含めて、OpenAI API に送信されます。
"""

openai.api_key = os.getenv("OPENAI_API_KEY")
slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
slack_app_token = os.getenv("SLACK_APP_TOKEN")


slack_client = WebClient(token=slack_bot_token)
bot_id = slack_client.auth_test()["user_id"]
