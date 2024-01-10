import telebot
from openai import OpenAI
from vk_bot.toxik import toxik_bot
client = OpenAI()

# Замените 'YOUR_BOT_TOKEN' на токен, который вы получили при регистрации бота в Telegram
TOKEN = ''

def prRed(skk):
  print("\033[91m {}\033[00m" .format(skk))

# Создайте объект бота
bot = telebot.TeleBot(TOKEN)

# Вывод инфирмации в консоль
def print_mesages(user, message_user, gen_text):
  print(f"{user}: {message_user}")
  print("BOT: " + gen_text)


# Обработчик текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_text(message):

    chat_id = message.chat.id
    user_name = message.chat.username
    mess = message.text

    if "/токсик" in mess and message.reply_to_message and message.reply_to_message.text is not None:     
      bot.send_message(message.chat.id,toxik_bot(message.reply_to_message.text))

    else:
      bot.send_message(message.chat.id,toxik_bot(message.text))
      pass

    


# Запустить бота
#bot.polling(none_stop=True, timeout=60)
bot.infinity_polling()