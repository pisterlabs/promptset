import random

from flask_mail import Message
from apps import App
import string

import openai
import time
from datetime import datetime

# class Utilities:
#     app = App()
#     mail = app.mail
#     mongo = app.mongo

#     def send_email(self, email):
#         msg = Message()
#         msg.subject = "BURNOUT - Reset Password Request"
#         msg.sender = 'bogusdummy123@gmail.com'
#         msg.recipients = [email]
#         random = str(self.get_random_string(8))
#         msg.body = 'Please use the following password to login to your account: ' + random
#         self.mongo.db.ath.update({'email': email}, {'$set': {'temp': random}})
#         if self.mail.send(msg):
#             return "success"
#         else:
#             return "failed"

#     def get_random_string(self, length):
#         # choose from all lowercase letter
#         letters = string.ascii_lowercase
#         result_str = ''.join(random.choice(letters) for i in range(length))
#         print("Random string of length", length, "is:", result_str)
#         return result_str


# Function to complete chat input using OpenAI's GPT-3.5 Turbo
def chatcompletion(user_input, impersonated_role, explicit_input,
                   chat_history):
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=2000,
        messages=[
            {
                "role":
                "system",
                "content":
                f"{impersonated_role}. Conversation history: {chat_history}"
            },
            {
                "role": "user",
                "content": f"{user_input}. {explicit_input}"
            },
        ])

    for item in output['choices']:
        chatgpt_output = item['message']['content']

    return chatgpt_output


# Function to handle user chat input
def chat(chat_history, name, chatgpt_output, user_input, history_file,
         impersonated_role, explicit_input):
    current_day = time.strftime("%d/%m", time.localtime())
    current_time = time.strftime("%H:%M:%S", time.localtime())
    chat_history += f'\nUser: {user_input}\n'
    chatgpt_raw_output = chatcompletion(user_input, impersonated_role,
                                        explicit_input,
                                        chat_history).replace(f'{name}:', '')
    chatgpt_output = f'{name}: {chatgpt_raw_output}'
    chat_history += chatgpt_output + '\n'
    with open(history_file, 'a') as f:
        f.write('\n' + current_day + ' ' + current_time + ' User: ' +
                user_input + ' \n' + current_day + ' ' + current_time + ' ' +
                chatgpt_output + '\n')
        f.close()
    return chatgpt_raw_output


# Function to get a response from the chatbot
def get_response(chat_history, name, chatgpt_output, userText, history_file,
                 impersonated_role, explicit_input):
    return chat(chat_history, name, chatgpt_output, userText, history_file,
                impersonated_role, explicit_input)


def calc_bmi(weight, height):
    return round((weight / ((height / 100)**2)), 2)


def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 24.9:
        return 'Normal Weight'
    elif bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obese'


def get_entries_for_email(db, email, start_date, end_date):

    # Query to find entries for a given email within the date range
    query = {'email': email, 'date': {'$gte': start_date, '$lte': end_date}}

    # Fetch entries from MongoDB
    entries_cal = db.calories.find(query)
    entries_workout = db.workout.find(query)

    return list(entries_cal), list(entries_workout)


def total_calories_to_burn(target_weight: int, current_weight: int):
    return int((target_weight - current_weight) * 7700)


def calories_to_burn(target_calories: int, current_calories: int,
                     target_date: datetime, start_date: datetime):
    actual_current_calories = current_calories - (
        (datetime.today() - start_date).days * 2000)

    new_target = target_calories - actual_current_calories

    days_remaining = (target_date - datetime.today()).days
    return int(new_target / days_remaining)
