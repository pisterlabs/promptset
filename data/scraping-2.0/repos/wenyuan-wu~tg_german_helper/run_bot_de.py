import os
import telebot
from openai_res import get_response_openai, get_response_openai_test
from dotenv import load_dotenv


def run_tg_bot(bot_token):
    bot = telebot.TeleBot(bot_token)

    @bot.message_handler(commands=['start'])
    def send_welcome(message):
        bot.reply_to(message, "Servus, wie geht's dir? Schreib mir in irgendeiner Sprache und ich helfe dir mit der "
                              "Übersetzung ins Deutsche!")

    @bot.message_handler(commands=['help'])
    def send_welcome(message):
        bot.reply_to(message, "Wenn das Ergebnis manchmal nicht zufriedenstellend ist, versuche es einfach ein "
                              "andermal. Die Flexibilität des Modells ist hoch, jede Ausgabe sollte unterschiedlich "
                              "sein, auch wenn die Eingabe die gleiche ist.")

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        reply = get_response_openai(message.text)
        bot.reply_to(message, reply)

    bot.infinity_polling()


def main():
    load_dotenv()
    bot_token = os.environ.get('BOT_TOKEN')
    run_tg_bot(bot_token)


if __name__ == '__main__':
    main()
