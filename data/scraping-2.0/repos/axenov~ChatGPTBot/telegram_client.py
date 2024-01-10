import json
import os
import urllib3
import random

from openai_client import openaiClient
from dinamodb_client import dynamoDBClient

BOT_ID = int(os.environ.get('BOT_ID'))
BOT_NAME = os.environ.get('BOT_NAME')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
FREQUENCY = float(os.environ.get('FREQUENCY'))
ALLOWED_CHATS = [int(num) for num in os.environ.get('ALLOWED_CHATS').split(',')]
RESET_COMMAND = os.environ.get('RESET_COMMAND')
CONTEXT_LENGTH = int(os.environ.get('CONTEXT_LENGTH'))

SEND_MESSAGE_URL = 'https://api.telegram.org/bot' + TELEGRAM_TOKEN + '/sendMessage'
http = urllib3.PoolManager()

dynamoDB_client = dynamoDBClient()
openai_client = openaiClient(dynamoDB_client)

class telegramClient:
    def __init__(self) -> None:
        pass
    
    def send_message(self, text: str, chat_id, original_message_id):
        """ Reply to a message of a user

        Args:
            text (str): the bot's message
            chat_id (int): id of a chat
            original_message_id (int): id of a message to reply to
        """
        payload = {
            "chat_id": chat_id,
            "parse_mode": "HTML",
            "text": text,
            "reply_to_message_id": original_message_id
        }
        response = http.request('POST', SEND_MESSAGE_URL, 
                                headers={'Content-Type': 'application/json'},
                                body=json.dumps(payload), timeout=10)
        print(response.data)
    
    def should_reply(self, message:dict):
        """ The function that decides whether the bot should reply to a message or not """
        if (
            (message["from"]["id"] == message["chat"]["id"]) or
            ("reply_to_message" in message and message["reply_to_message"]["from"]["id"] == BOT_ID) or
            ("entities" in message and message["entities"][0]["type"] == "mention" and ("@" + BOT_NAME) in message["text"])
        ):
            return True
        else:
            bet = random.random()
            if bet < FREQUENCY:
                return True
        return False
    
    def process_message(self, body):
        """ Process a message of a user and with some probability reply to it

        Args:
            body (str): a telegram webhook body
        """
        if (
            "message" in body and
            not body["message"]["from"]["is_bot"] and
            "forward_from_message_id" not in body["message"]
            ):
            message = body["message"]
            
            chat_id = message["chat"]["id"]
            message_id=message["message_id"]
            if chat_id not in ALLOWED_CHATS:
                return

            if "entities" in message and message["entities"][0]["type"]  == "bot_command" and  ("/" + RESET_COMMAND) in message["text"]:
                dynamoDB_client.reset_chat(f"{str(chat_id)}_{str(BOT_ID)}")
                return
            
            # Extract the message of a user
            if "text" in body["message"]:
                user_message = message["text"]
            elif "sticker" in body["message"] and "emoji" in body["message"]["sticker"]:
                user_message = message["sticker"]["emoji"]
            elif "photo" in body["message"] and "caption" in body["message"]:
                user_message = message["caption"]
            else:
               return

            if self.should_reply(message):
                user_message = user_message.replace("@" + BOT_NAME, "")
                bot_message = openai_client.complete_chat(user_message, chat_id, BOT_ID)
                self.send_message(bot_message, chat_id, message_id)
            else:
                previous_messages = dynamoDB_client.load_messages(f"{str(chat_id)}_{str(BOT_ID)}")[-CONTEXT_LENGTH:]
                dynamoDB_client.save_messages(f"{str(chat_id)}_{str(BOT_ID)}", previous_messages[-(CONTEXT_LENGTH-1):] + [{"role": "user", "content": user_message}])