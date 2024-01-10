# Telegram Bot to communicate with OpenAI API
import logging
import telebot
import openai


logger = telebot.logger
telebot.logger.setLevel(logging.DEBUG)

load_dotenv()
bot = telebot.TeleBot("put your bot token")
openai.api_key = "chatgpt api key"
user_id = int("chat id") #only for private messaging


@bot.message_handler(func=lambda message: True)
def get_response(message):
  if int(message.chat.id) != user_id:
    bot.send_message("This bot is not for public but private use only.")
  else:
    response = ""
    if message.text.startswith(">>>"):
      # Use Codex API for code completion
      response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f'```\n{message.text[3:]}\n```',
        temperature=0,
        max_tokens=4000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n", ">>>"],
      )
    else:
      # Use GPT API for text completion
      # Check if the question is about code or not
      if "code" in message.text.lower() or "python" in message.text.lower():
        # Use Codex API for code-related questions
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=f'"""\n{message.text}\n"""',
          temperature=0,
          max_tokens=4000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=['"""'],
        )
      else:
        # Use GPT API for non-code-related questions
        response = openai.Completion.create(
          engine="text-davinci-003",
          prompt=f'"""\n{message.text}\n"""',
          temperature=0,
          max_tokens=2000,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=['"""'],
        )

    bot.send_message(message.chat.id, f'{response["choices"][0]["text"]}', parse_mode="None")

bot.infinity_polling()







# import telebot
# import openai

# # Set up the OpenAI API
# openai.api_key = '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'

# # Set up the Telegram bot
# bot = telebot.TeleBot('6067516516:AAGaRCO8LSxI2RpU5tW8I0OklVwURJIXAVM')

# # Define a function to handle incoming messages
# @bot.message_handler(func=lambda message: True)
# def handle_message(message):
#     response = openai.Completion.create(
#         engine='davinci', prompt=message.text, max_tokens=1024, n=1, stop=None, temperature=0.7
#     )
#     text = response.choices[0].text.strip()
#     bot.send_message(message.chat.id, text)

# # Start the Telegram bot
# bot.polling()
