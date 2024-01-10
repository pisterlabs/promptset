# System Imports
import os
from shlex import quote as os_quote
from typing import Union

# Package Imports
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
FIRST_OUTPUT = "Hello, I am a chat bot. Ask me anything."
FILE = open("debug_log.txt", "w")
TTS_ENABLED = os.getenv("TTS_ENABLED")

# Init chat log
chat_log = f"{FIRST_OUTPUT}\n"

# Init openai interface
openai.api_key = os.getenv("OPENAI_API_KEY")
completion = openai.Completion()


def openai_respond(text: str, chat_log: str) -> Union[str, str]:

    # get response from openai interface
    prompt = f"{chat_log}Human: {text}\n"
    response = completion.create(
        prompt=prompt,
        engine="davinci",
        stop=["\nHuman"],
        temperature=0.9,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        best_of=1,
        max_tokens=150,
    )

    return [response.choices[0].text.strip(), chat_log]


def respond(text: str, chat_log: str) -> Union[str, str]:
    if text == "hi":
        return ["hello, how are you?", chat_log]
    elif text == "bye":
        return ["goodbye", chat_log]
    else:
        # if response not hard coded, get the chatbot to respond
        return openai_respond(text, chat_log)


def output(text: str) -> None:
    print("\n" + text)

    # check if TTS enabled, if so, speak text
    if TTS_ENABLED:
        os.system(f"say {os_quote(text)}")


output(FIRST_OUTPUT)

while True:
    # get some text from the user
    print("Enter some text: ")
    text = input()

    # get a response
    response_arr = respond(text, chat_log)
    response_text = response_arr[0]
    chat_log = response_arr[1]

    # output the response
    output(response_text)

    chat_log = f"{chat_log}Human: {text}\n{response_text}\n"
    FILE.write(chat_log)

    if text == "bye":
        FILE.close()
        break
