import os
from openai import OpenAI
from dotenv import load_dotenv
import re

load_dotenv("../.env")

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

full_conversation = []

run_counter = 0


def gpt_call(user_message, system_prompt):
    global full_conversation

    global run_counter
    run_counter += 1

    # set the initial prompt to get the theme from GPT3
    if run_counter == 1:
        full_conversation = [{"role": "system", "content": system_prompt}]
    # Set the ongoing system prompt to include the response of the theme
    if run_counter == 2:
        full_conversation = [{"role": "system", "content": system_prompt}]
    # append the new user message to the conversation
    full_conversation.append({"role": "user", "content": user_message})

    chat_completion = client.chat.completions.create(
        messages=full_conversation,
        model="gpt-3.5-turbo",
        temperature=0.6,
        top_p=1,
        max_tokens=150,
    )
    # add the response to the conversation
    assistant_message = {
        "role": "assistant",
        "content": chat_completion.choices[0].message.content,
    }
    full_conversation.append(assistant_message)
    # print(full_conversation)
    # return the response
    return chat_completion.choices[0].message.content


# Initial prompt to get the theme (build some tension)
user_msg = input("The text prompt flashes awaiting your input. \n")

# initial theme system message
theme_system_msg = "You choose a theme for a choose your own adventure game. Return only the theme in a short concise sentence. Always respond with a theme."
# call the gpt_call function to get the theme
game_theme = gpt_call("Please give me a mysterious theme.", theme_system_msg)
# includ the theme in the system message
intro_system_msg = (
    "You are a choose your own adventure dungeon master. You take the user's input and create a part of a story based on their choices, you always finish your response with a question unless the user is dead. You then ask them if they would like to play again. The story is created based on the user's input and the theme of "
    + game_theme
    + ". When the user dies, return 'YOU ARE DEAD'. NEVER response with As an AI"
)
play_again = "yes"


def game():
    global play_again
    play_again = "no"
    global user_msg
    while True:
        last_response = gpt_call(user_msg, intro_system_msg)
        formatted_text = re.sub(r"([.?!])(\s|$)", r"\1\n", str(last_response))
        if "YOU ARE DEAD" in last_response:
            play_again = input("Would you like to play again? \n").lower()
            break
        print("\n")
        user_msg = input(str(formatted_text))
        print("\n \n")


if play_again == "yes":
    game()
