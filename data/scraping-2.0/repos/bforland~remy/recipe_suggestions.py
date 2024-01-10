def clear():
    print("\033[J\033[K")


clear()

import pandas as pd
import numpy as np

import openai
import time

menu = pd.read_csv("./data/sample_upcoming_data.csv")
menu["title_split"] = menu["title"].apply(
    lambda s: s.replace("[", "/").replace("]", "/").split("/")
)
menu["title_len"] = menu["title_split"].apply(len)
menu = menu.query("title_len==3")
menu["title"] = menu["title_split"].apply(lambda l: l[2].lower())
menu = menu.drop(["title_split", "title_len"], axis=1)
menu = menu.iloc[:3]

openai.api_key = "YOUR_API_KEY_HERE"


def chatbot(conversation_history):

    mess_content = """
    You're incredibly personable.
    You are an AI assistant for HelloFresh that will help customers interacting with the menu.
    Don't ask for the customers email and account information.
    Keep replies short and concise.
    Refer to yourself in the first person.
    When the conversation ends say "have a great day".
    It is important to establish if the customer wants to continue receiving a delivery after the pause period.
    """
    mess_content += "You are also abot to make recommendation for HelloFresh meal."
    mess_content += "It is important to keep in mind, you can only make the suggestion from the following meals:"

    for title in list(menu["title"]):
        mess_content += title + ","
    cuisine = list(menu["cuisine"])
    for idx, title in enumerate(list(menu["title"])):
        s = "the cuisine of" + title + f" is {cuisine[idx]},"
        # print (s)
        mess_content += s

    protein = list(menu["primary_protein"])
    for idx, title in enumerate(list(menu["title"])):
        s = "the protein of" + title + f" is {protein[idx]},"
        # print (s)
        mess_content += s

    mess_content += "Every time only recommend 3 meals at most."
    mess_content += "Do not mention the cuisine and protein of the food unless the customer asks for it."
    mess_content += (
        "If the customer ask what type the food is, answer it with the cusine."
    )
    mess_content += "If the customer ask for a food, \
        match it with the protein of the food first."
    mess_content += "Do not finish the sencentence with a question."

    # print (mess_content)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": mess_content}, *conversation_history],
    )
    return response["choices"][0]["message"]["content"]


history = []
u_input = [
    "Hello there, could you help me with hellofresh menu?",
    "Yes. Could you recommend some hellofresh menu to me?",
    "Got it. Do you have any beef?",
    "I see. What type the foold is?",
    "I just had this type of food last week, looking for something different.Any lobster in HelloFresh menu?",
    "So sad. Any chicken?",
    "I have had all of them. Probably I will just skip HelloFresh this week. Can you help with pausing HelloFesh?",
]

b_output = []
for i in range(len(u_input)):
    time.sleep(10)

    user_input = u_input[i]
    print("\nUser:", user_input)
    # Start recorder with the given values
    history.append({"role": "user", "content": user_input})
    bot_output = chatbot(history)
    print("\nBot:", bot_output)

    history.append({"role": "assistant", "content": bot_output})


# print (history)
