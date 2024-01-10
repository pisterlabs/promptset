import telebot
import openai
import os

# Set up OpenAI API credentials
openai.api_key = "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"

# Set up Telegram bot token
token = "6252170415:AAGAbum5zUJwprkuFZSjV_CKZKvIp9cwSp8"
bot = telebot.TeleBot(token)

# Define handler for incoming messages
@bot.message_handler(func=lambda message: True)
def handle_message(message):
    # Get user input
    user_input = message.text

    # Call OpenAI's GPT-3 to generate a response
    prompt = f"Conversation with user:\nUser: {user_input}\nChatGPT:"
    response = openai.Completion.create(engine="davinci", prompt=prompt, max_tokens=1024)["choices"][0]["text"].strip()

    # Send response back to user
    bot.reply_to(message, response)

# Start the bot
bot.polling()
