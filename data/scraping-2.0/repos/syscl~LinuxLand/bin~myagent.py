#!/usr/bin/env python3

# This is a chat bot like CLI aims for getting answers from chat in a loop
# Updates will include any models for it (TBD)

import os
from dotenv import load_dotenv, find_dotenv
import logging
import sys
import readline

from openai import OpenAI

load_dotenv(find_dotenv())
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

log_format = "[%(levelname)s][%(asctime)s][%(funcName)s] %(message)s"
logging.basicConfig(level=logging.WARNING, format=log_format)
logger = logging.getLogger(__name__)


VERSION = "0.1.6"
# Make this configurable
GPT_MODEL_VERSION = "gpt-4"


def main():
    print(f"AI Agent v{VERSION}, model version: {GPT_MODEL_VERSION}")
    print("Type help for more commands in the console")

    system_message = {"role": "system", "content": "You are a super intelligent assistant."}
    messages = [system_message]
    while True:
        try:
            # library readline will handle the input (via `input()`)
            # like bash terminal. Alternative will be `prompt_toolkit` library
            # TODO(syscl): handle arrow keys up and down for history
            message = input("> ")
            strip_message = message.strip().lower()
            # Ensure it is not an empty string
            if strip_message:
                if strip_message == "clear":
                    messages = [system_message]
                elif strip_message == "help":
                    print(
                        "command: clear for starting a new chat without history (context)"
                    )
                elif strip_message == "exit":
                    sys.exit(0)
                else:
                    messages.append(
                        {"role": "user", "content": message},
                    )
                    chat = client.chat.completions.create(
                        model=GPT_MODEL_VERSION, messages=messages
                    )
                    reply = chat.choices[0].message.content
                    print(f"AI: {reply}\n")
                    messages.append({"role": "assistant", "content": reply})
        except KeyboardInterrupt:
            messages = [system_message]
            print("KeyboardInterrupt, clear all previous context.")
        except EOFError:
            # Hide the "^D" on the terminal
            print("" * len("^D"))
            sys.exit(0)


if __name__ == "__main__":
    main()
