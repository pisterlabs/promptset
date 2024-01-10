#!/usr/bin/env python3

import os 
import openai # OpenAI API import
import logging
import requests
from telegram import Update, InputMediaPhoto
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import tiktoken 


# OepnAI API Key goes here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Telegram Bot Token goes here
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_API_KEY")

#Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# Enable basic logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
# End of logging

# Initialize conversation history
conversation_history = []

# Initialize TikTok Token -- used for encoding the prompt
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))

# Start the bot function -- runs when /start is called and automatically sends .reply_text to the user in a personal chat
def start(update: Update, context: CallbackContext) -> None:
    update.message.reply_text('Hi! I am a ChatGPT bot. Send me a message and I will try to respond intelligently.')

# GPT chat function -- runs when 'gpt' handler is called and automatically sends return to the user in a public chat
def gpt_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003", # Insert engine/model here
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=1.0,
        frequency_penalty=1.0,
    )
    return response.choices[0].text.strip()

def chat(update: Update, context: CallbackContext) -> None:
    user_message = update.message.text
    prompt = user_message
    response = gpt_response(prompt)
    update.message.reply_text(f"Davinci Model: {response}")

def handle_message(update: Update, context: CallbackContext):

    if update.message and update.message.text:
        user_message2 = update.message.text

    global conversation_history
    user_message2 = update.message.text

    # Check if the user's message is too long -- needs to be optimized
    if count_tokens(user_message2) > 1000:
        print("Your prompt is too long. Please shorten and try again")
    else:
        try:
            # Add user message to the conversation history
            conversation_history.append({"role": "user", "content": user_message2})

            # Call the OpenAI API with the chat.create endpoint
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=conversation_history,
                max_tokens=1000,
                n=1,
                temperature=0.6,
                stop="content"
            )

            # Extract the assistant's response  
            assistant_message = response.choices[0].message["content"]

            # Add assistant message to the conversation history
            conversation_history.append({"role": "assistant", "content": assistant_message})

            # Send the assistant's response to the user
            update.message.reply_text(assistant_message)
        except openai.Error as e:
            if e.args[0]['error']['code'] == 'context_length_exceeded':
                print("Error: Context length exceeded. Please reduce the input text.")
            else:
                print("Error: ", e)
            return None


# DALL-E image generation function 
def generate_image(prompt):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        'model': 'image-alpha-001',
        'prompt': prompt,
        'num_images': 1,
        'size': '256x256',
        'response_format': 'url'
    }

    response = requests.post('https://api.openai.com/v1/images/generations', json=data, headers=headers)
    response_json = response.json()

    if response.status_code == 200:
        return response_json['data'][0]['url']
    else:
        raise Exception(f"Error generating image: {response_json['message']}, please try your prompt again")

def generate_and_send_image(update: Update, context: CallbackContext, prompt: str) -> None:
    try:
        image_url = generate_image(prompt)
        update.message.reply_photo(photo=image_url)
    except Exception as e:
        update.message.reply_text(f"Error generating image: {str(e)}, please try your prompt again")

def image_request(update: Update, context: CallbackContext) -> None:
    prompt = update.message.text
    generate_and_send_image(update, context, prompt)


# main function
def main():
    updater = Updater(TELEGRAM_BOT_TOKEN)

    dispatcher = updater.dispatcher
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("gpt", handle_message))
    dispatcher.add_handler(CommandHandler("gptd", chat))
    dispatcher.add_handler(CommandHandler("dalle", image_request))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()