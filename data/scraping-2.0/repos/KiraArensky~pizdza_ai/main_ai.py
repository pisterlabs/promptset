# импорт библиотек
import os

try:
    import cowsay
    import telebot
    import openai
except ModuleNotFoundError:
    os.system("pip install cowsay")
    os.system("pip install openai")
    os.system("pip install pyTelegramBotAPI")
    import telebot
    import cowsay
    import openai

import time

# токен бота из BotFather
bot = telebot.TeleBot('YOUR_TG_BOT_API_KEY')


# функция для chatgpt
def ai_request(request_mess):
    # токен вашего бота и OpenAI API key из секретного менеджера OpenAI
    openai.api_key = "YOUR_OPENAI_API_KEY"

    completion = openai.ChatCompletion.create(  # Запрос
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": request_mess}  # Сообщение для GPT
        ]
    )
    return completion.choices[0].message.content


# это прикол, чтобы проверить запустился ли код
print(cowsay.get_output_string('pig', "Бот запущен!"))  # это для проверки, что бот запустился
print(" " * 10, "##" * 12)
print("", end='\n')


# запуск бота
@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, text='Данный бот для связи с chatgpt-turbo-3.5\n\nВведите запрос в формате '
                                  '"/ai [ваш запрос]"\n'
                                  '\nОтец: @pizdza_mnyam')


# включили ии
@bot.message_handler(content_types=['text'])
def ai_on(message):
    if str(message.text).lower() == "/ai":
        msg = bot.send_message(message.chat.id, text="Жду вашего запроса")

        bot.register_next_step_handler(msg, ai_answer)

    elif str(message.text[:3]).lower() == "/ai":
        # обманка
        msg = bot.send_message(message.chat.id, text="Ждите, запрос обрабатывается")

        # ответ от chatgpt
        answer_ai = ai_request(message.text)

        # отправка сообщения пользователю
        time.sleep(1)
        bot.edit_message_text(chat_id=message.chat.id, message_id=msg.message_id, text=f'{answer_ai}')


def ai_answer(message):
    msg = bot.send_message(message.chat.id, text="Ждите, запрос обрабатывается")

    # ответ от chatgpt
    answer_ai = ai_request(message.text)

    # отправка сообщения пользователю
    time.sleep(1)
    bot.edit_message_text(chat_id=message.chat.id, message_id=msg.message_id, text=f'{answer_ai}')


# ожидание взаимодейтвия
if __name__ == '__main__':
    bot.polling(none_stop=True, interval=0)
