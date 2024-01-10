import os
import html
import openai
import requests
from datetime import date
from abc import ABC, abstractmethod
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext, ConversationHandler
from common import get_lobby_keyboard
from commands import Command

openai.api_key = os.environ.get('OPEN_AI_KEY')

class Stories(Command):
    def execute(self, update: Update, context: CallbackContext) -> None:
        city_name = self.get_city_name(context)
        update.message.reply_text(f"Here are some facts about {city_name}:")
        city_facts = self.get_facts(city_name)
        print('city', city_facts)
    
    def get_facts(self, city_name: str) -> str:
        prompt = f"Tell me some interesting facts about {city_name}"
        
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt
            )
        except Exception as e:
            print(f"An error occured: {e}")
        
        facts = response.choices[0].text.strip()
        return facts
    
    def get_city_name(self, context: CallbackContext) -> str:
        city_data = context.user_data.get('city_data')[0]
        address_components = city_data.get('address_components')[0]
        city_name = address_components.get('long_name')
        return city_name