import openai
import sounddevice as sd
from scipy.io.wavfile import write

# import wavio as wv
import os
import datetime as dt

# import numpy as np
from gtts import gTTS

openai.api_key = "YOUR-KEY-HERE"


def status_check():

    customer_status = chatbot(
        [{"role": "user", "content": "What is my subscription status right now ?"}]
    )

    if "paused" in customer_status.lower():
        customer_status = "Paused"
    elif "active" in customer_status.lower():
        customer_status = "Active"
    else:
        customer_status = "Unknown"

    return customer_status


def get_hf_week_from_chatbot(my_string):
    my_string = "the year and week number for the weeks you are paused are as follows: - week 25 (june 19th - june 25th, 2023) - week 26 (june 26th - july 2nd, 2023) - week 27 (july 3rd - july 9th, 2023)"
    my_string = my_string.split(":")[1].split(")")
    for val in my_string:
        week = ""
        year = ""
        if ", " in val:
            year = val.split(", ")[1]
            week = val.split(" (")[0].split("week ")[1]
            print(str(year) + "-W" + str(week))


def chatbot(conversation_history):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "assistant",
                "content": f"""
            You're incredibly personable.
            You are an AI assistant for HelloFresh that will help customers interacting with the menu.
            Don't ask for the customers email and account information.
            Keep replies short and concise.
            Refer to yourself in the first person.
            You are able to help customers pause their subscriptions for a specified number of weeks.
            When the conversation ends say "have a great day".
            It is important to establish if the customer wants to continue receiving a delivery after the pause period.
            You can change the global variable customer_status and should change it baeed on the conversation.
            For the next five weeks the customer status is: Active, Active, Active, Active, Active.
            Customers can ask for it to be "Paused". Change to "Paused" if requested based on the conversation.
            Change all number works to digits. Never ask the customer for to cancel.
            Assume the customer wants to continue receiving a delivery after the pause period.
            Today's date is June 21th, 2023.
            """,
            },
            *conversation_history,
        ],
        temperature=0.1,
    )
    return response["choices"][0]["message"]["content"]


history = []

while True:
    user_input = input("User: ")

    if ("Goodbye").lower() in user_input.lower():
        break

    history.append({"role": "user", "content": user_input})

    bot_output = chatbot(history)
    print("Bot:", bot_output)
    # text_to_speech(bot_output)
    # os.system(f"say {bot_output}")
    history.append({"role": "assistant", "content": bot_output})

# history.append({"role": "user", "content": "WHat is the integer number of weeks I'm paused for, only the number please ?"})
# num_weeks = int(chatbot(history).lower().split('weeks')[0].split(' ')[-2])
# print(num_weeks)

history = history[:-1]
history.append(
    {
        "role": "user",
        "content": "What is the year and week number for the weeks I'm paused ?",
    }
)
get_hf_week_from_chatbot(chatbot(history).lower())
