import wikipedia
import webbrowser
# import os
import random
import string
import re
import datetime
import openai

class Chatgpt:
    def __init__(self) -> None:
        openai.api_key = 'your_chatgpt_api_key'

    def write_code(self, prompt_text):
        response = openai.Completion.create(
            model="code-davinci-002",
            prompt=prompt_text,
            temperature=0,
            max_tokens=256,
        )

        return response

    def text_out(self, prompt_text: str):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_text,
            max_tokens=400,
            temperature=0.9,
        )
        return response

    def generate_image(self, prompt_text):
        response = openai.Image.create(
            prompt=prompt_text,
            n=3,
            size="1024x1024"
        )
        return response
# password_generator

def passwordgenerator(passlen=8):
    """This is normal password generator which help to generate password.
    Digits, special letters, uppercase, lowercase letters are use to generate
    password."""

    s1 = string.ascii_letters
    s2 = string.digits
    s3 = string.punctuation
    # print(s3)
    s = []
    s.extend(list(s1))
    s.extend(list(s2))
    s.extend(list(s3))
    return ("".join(random.sample(s, passlen)))

chat = Chatgpt()
def get_response(message: str) -> str:
    command_user = message.lower()

    if 'wikipedia' in command_user:
        command_user = command_user.replace("wikipedia", "")
        results = wikipedia.summary(command_user, sentences=2)
        return results

    elif 'open youtube' in command_user:
        webbrowser.open("youtube.com")

    elif 'open google' in command_user:
        webbrowser.open("google.com")

    elif 'open stackoverflow' in command_user:
        webbrowser.open("stackoverflow.com")

    elif 'open amazon' in command_user:
        webbrowser.open("amazon.com")

    elif 'open flipkart' in command_user:
        webbrowser.open("flipkart.com")

    elif 'open whatsapp' in command_user:
        webbrowser.open("whatsapp.com")

    elif 'the time' in command_user:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        return (f"Sir, the time is {strTime}")

    elif 'random password' in command_user:
        try:
            length = int(re.search(r'\d+', command_user).group())

        except AttributeError:
            length = 8

        password = passwordgenerator(int(length))
        return (f"Your password is `{str(password)}`")

    elif 'bye' in command_user or 'stop' in command_user or 'exit' in command_user:
        return "Bye Sir, have a good day."
        exit()

    elif 'generate image' in command_user:

        response = chat.generate_image(prompt_text=command_user)
        for image in response['data']:
            return image['url']


    elif 'write code' in command_user or 'write a code' in command_user or 'write program' in command_user:
        response = chat.write_code(prompt_text=command_user)

        for choice in response['choices']:
            return choice['text']

    else:
        response = chat.text_out(prompt_text=command_user)

        for choice in response['choices']:
            return choice['text']