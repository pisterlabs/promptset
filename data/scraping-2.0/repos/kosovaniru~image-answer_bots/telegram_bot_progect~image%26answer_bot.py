import cohere
import telebot
import requests
import os
import base64
from dotenv import load_dotenv, find_dotenv
from telebot import types
import threading
import time

load_dotenv(find_dotenv())


class ImageBot:
    def __init__(self):
        self.tg_bot_image = telebot.TeleBot(os.getenv('TG_IMAGE_BOT'))
        self.api_host = os.getenv('API_HOST')
        self.api_key = os.getenv('API_KEY')
        self.engine_id = os.getenv('ENGINE_ID')

        @self.tg_bot_image.message_handler(commands=['start'])
        def start(message):
            self.tg_bot_image.send_message(message.chat.id, f'Hi, {message.from_user.first_name}! What do you want to generate?')

        @self.tg_bot_image.message_handler(commands=['help'])
        def to_help(message):
            self.tg_bot_image.send_message(message.chat.id,'Hello! I am MustiCanvasBot, a bot that will help you generate photos. All you need to do is just tell me what you want, and I will try to generate it!')

        @self.tg_bot_image.message_handler(content_types=['text'])
        def get_photo(message):
            message_user = message.text.lower()
            response = requests.post(f"{self.api_host}/v1/generation/{self.engine_id}/text-to-image", headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }, json={
                "text_prompts": [
                    {
                        "text": message_user
                    }
                ],
                "cfg_scale": 7,
                "clip_guidance_preset": "FAST_BLUE",
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 30,
            })

            data = response.json()

            for i, image in enumerate(data["artifacts"]):
                with open(f"./text_to_image_{i}.png", "wb") as f:
                    f.write(base64.b64decode(image["base64"]))

            with open(f"./text_to_image_{i}.png", "rb") as photo:
                markup = types.InlineKeyboardMarkup()
                btn1 = types.InlineKeyboardButton('Delete image', callback_data='delete')
                markup.row(btn1)
                self.tg_bot_image.send_photo(message.chat.id, photo, reply_markup=markup)

        @self.tg_bot_image.callback_query_handler(func=lambda callback: True)
        def delete(callback):
            if callback.data == 'delete':
                self.tg_bot_image.delete_message(callback.message.chat.id, callback.message.message_id - 0)

    def run_polling(self):
        self.tg_bot_image.polling(none_stop=True)

class AnswerBot:
    def __init__(self):
        self.tg_bot_answer = telebot.TeleBot(os.getenv('TG_ANSWER_BOT'))
        self.coherent = cohere.Client(os.getenv('COHERENT'))

        @self.tg_bot_answer.message_handler(commands=['start'])
        def send_hello(message):
            self.tg_bot_answer.send_message(message.chat.id, f'Hi, {message.from_user.first_name}! What is your question?')

        @self.tg_bot_answer.message_handler(commands=['help'])
        def info(message):
            self.tg_bot_answer.send_message(message.chat.id,
                'Hello! I am InsightIQBot, a bot that will help you get answers to your questions! Just write your question to me, and I will generate an answer for you!"')

        @self.tg_bot_answer.message_handler(content_types=['text'])
        def get_info(message):
            text_from_user = message.text.lower()
            response = self.coherent.chat(
                text_from_user,
                model="command",
                temperature=0.9
            )

            answer = response.text
            self.tg_bot_answer.send_message(message.chat.id, answer)

    def run_polling(self):
        self.tg_bot_answer.polling(none_stop=True)

if __name__ == '__main__':
    image_bot = ImageBot()
    answer_bot = AnswerBot()

    image_thread = threading.Thread(target=image_bot.run_polling)
    answer_thread = threading.Thread(target=answer_bot.run_polling)

    image_thread.start()
    answer_thread.start()

    image_thread.join()
    answer_thread.join()







