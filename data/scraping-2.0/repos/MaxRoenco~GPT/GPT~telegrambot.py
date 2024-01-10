import telebot
import openai

telegram_key = '6782572542:AAEk_oSfzGq6nRjo4PJf95nmVtAlyZef1Ko'
openai.api_key = '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'

bot = telebot.TeleBot(telegram_key)

@bot.message_handler(commands=['start'])
def hello(message):
    bot.send_message(message.chat.id, 'Привет, я твой ChatGPT бот. Готов тебе помочь!')

@bot.message_handler(content_types=['text'])
def main(message):
    reply = ''
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=message.text,
        max_tokens=300,
        temperature=0.7,
        n=1,
        stop=None
    )

    if response and response.choices:
        reply = response.choices[0].text.strip()
    else:
        reply = 'Ой, что-то не так!('

    bot.send_message(message.chat.id, reply)

bot.polling(none_stop=True)