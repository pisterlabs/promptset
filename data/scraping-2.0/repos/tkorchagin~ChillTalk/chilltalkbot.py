import openai
import telebot
from telebot import types
import random
import time

import config as co
import templates as t

openai.api_key = co.OPENAI_API_KEY
bot = telebot.TeleBot(co.TELEGRAM_BOT_TOKEN)

USERS_INFO = {}


def clean(s):
    s = ' '.join(s.split())
    return s


def get_next_message(text_to_process="", system_context="", message_history=[]):
    response = {}

    messages = []
    if system_context:
        messages += [{"role": "system", "content": system_context}]

    if message_history:
        messages += message_history

    if text_to_process:
        messages += [{"role": "user", "content": text_to_process}]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0, # this is the degree of randomness of the model's output
            presence_penalty=2,
        )
    except Exception as e:
        print(e)

    choices = response.get('choices', [])
    # print(response)

    if choices:
        next_message = choices[0]['message']['content']
    else:
        next_message = ''

    return next_message


def make_message_to_process(situation, formula_to_answer, user_answer):
    message_to_process = ""
    message_to_process += "<b>Ситуация:</b>"
    message_to_process += "\n\n"
    message_to_process += f"```\n{situation}\n```"
    message_to_process += "\n\n"

    message_to_process += "<b>Дайте обратную связь по формуле:</b>"
    message_to_process += "\n\n"
    message_to_process += f"```\n{formula_to_answer}\n```"
    message_to_process += "\n\n"

    message_to_process += "<b>Мой ответ:</b>"
    message_to_process += "\n\n"
    message_to_process += f"```\n{user_answer}\n```"

    return message_to_process


def generate_formula(techniques, management_impact):
    selected_techniques = random.sample(techniques, random.randint(2, 4))
    techniques_string = " → ".join(selected_techniques)
    management_impact_string = random.choice(management_impact)
    result = f"{techniques_string} → {management_impact_string}"
    return result


def get_situation(situations):
    situation = random.choice(situations)
    return situation


def send_new_task(chat_id, situations, techniques, management_impact):
    global USERS_INFO

    formula_to_answer = generate_formula(techniques, management_impact)
    situation = get_situation(situations)

    USERS_INFO[chat_id] = {
        'formula_to_answer': formula_to_answer,
        'situation': situation,
    }

    text_to_send = ""
    text_to_send += "<b>Ситуация:</b>"
    text_to_send += "\n\n"
    text_to_send += situation
    text_to_send += "\n\n"
    text_to_send += "<b>Дайте ответ по формуле:</b>"
    text_to_send += "\n\n"
    text_to_send += formula_to_answer

    print(chat_id, text_to_send)
    bot_send_message(chat_id, text_to_send, message=None)


def bot_send_message(chat_id, text_to_send, message=None):
    try:
        keyboard = types.ReplyKeyboardMarkup(row_width=1)
        keyboard.add(types.KeyboardButton(t.CHANGE_SITUATION_TEXT))
        keyboard.add(types.KeyboardButton(t.CHANGE_DECODING_TEXT))

        if message:
            bot.reply_to(message, text_to_send, reply_markup=keyboard, parse_mode='HTML')
        else:
            bot.send_message(chat_id, text_to_send, reply_markup=keyboard, parse_mode='HTML')
    except Exception as e:
        print(e)


@bot.message_handler(commands=['start'])
def handle_start(message):
    chat_id = message.chat.id
    bot_send_message(chat_id, t.START_TEXT)
    time.sleep(5)

    send_new_task(
        chat_id=chat_id,
        situations=t.SITUATIONS,
        techniques=t.TECHNIQUES,
        management_impact=t.MANAGEMENT_IMPACT,
    )


@bot.message_handler(func=lambda message: message.text.lower() == t.CHANGE_SITUATION_TEXT.lower())
def handle_change_dictionary(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')

    send_new_task(
        chat_id=chat_id,
        situations=t.SITUATIONS,
        techniques=t.TECHNIQUES,
        management_impact=t.MANAGEMENT_IMPACT,
    )


@bot.message_handler(func=lambda message: message.text.lower() == t.CHANGE_DECODING_TEXT.lower())
def handle_change_decoding_text(message):
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')
    text_to_send = random.choice(t.DECODING_SITUATIONS)
    print(chat_id, text_to_send)
    bot_send_message(chat_id, text_to_send, message=None)


@bot.message_handler(func=lambda message: True)
def message_handler(message):
    global USERS_INFO

    chat_id = message.chat.id
    bot.send_chat_action(chat_id, 'typing')

    print('#' * 80)
    print(chat_id, message.text)
    print()

    if chat_id not in USERS_INFO:
        send_new_task(
            chat_id=chat_id,
            situations=t.SITUATIONS,
            techniques=t.TECHNIQUES,
            management_impact=t.MANAGEMENT_IMPACT,
        )

    else:
        message_to_process = make_message_to_process(
            situation=USERS_INFO[chat_id].get('situation', ''),
            formula_to_answer=USERS_INFO[chat_id].get('formula_to_answer', ''),
            user_answer=message.text
        )
        bot_send_message(chat_id, t.WAIT_MESSAGE_TEXT, message=message)
        bot.send_chat_action(chat_id, 'typing')
        # bot_send_message(chat_id, message_to_process, message=message)

        answer_review = get_next_message(
            text_to_process=message_to_process,
            system_context=t.SYSTEM_SONTEXT,
            message_history=[],
        )

        bot_send_message(chat_id, answer_review, message=message)


if __name__ == '__main__':
    print('#started')
    
    while True:
        try:
            bot.polling()
        except Exception as e:
            print(e)
            time.sleep(5)

