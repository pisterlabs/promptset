import telebot
import openai
from telebot.apihelper import ApiTelegramException


# Initialize the telegram bot with a bot token
bot = telebot.TeleBot("")

# Set the API key for OpenAI
openai.api_key = ""


# Handle the /start command
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hi! I'm a bot powered by OpenAI. How can I help you today?")


# Handle all other messages
@bot.message_handler(func=lambda message: True)
def answer_question(message):
    # Get the user's question
    question = message.text

    try:
        # Use OpenAI API to generate an answer
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are are the best AI assistant."},
                {"role": "user", "content": question}
            ]
        )

        # Get the answer from the response
        answer = completion["choices"][0]["message"]["content"]

        # Send the answer back to the user
        bot.reply_to(message, answer)
    except ApiTelegramException as e:
        bot.reply_to(message, "An error occurred: " + str(e))


# Start the bot
bot.polling(none_stop=True, timeout=120)