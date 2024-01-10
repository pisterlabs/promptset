#!/usr/bin/env python3
import argparse
import asyncio
import importlib
import os
from collections import deque

import openai
from dotenv import load_dotenv
from telethon import TelegramClient, events

UWU_LIB_AVAIL = importlib.util.find_spec("uwuify") is not None
if UWU_LIB_AVAIL:
    import uwuify

# env only
load_dotenv()
OPENAI_KEY = os.environ.get("OPENAI_KEY")
TG_API_ID = os.environ.get("TG_API_ID")
TG_API_HASH = os.environ.get("TG_API_HASH")
MY_USER_ID = os.environ.get("MY_USER_ID")
if MY_USER_ID:
    MY_USER_ID = int(MY_USER_ID)

# args only
UWU = False

# resultants
openai.api_key = OPENAI_KEY
completion = openai.Completion()
client = TelegramClient("listener", TG_API_ID, TG_API_HASH)

PRESET_PROMPT = []

MAX_SAVED_MESSAGES = 10
ENGINE = "babbage"  # $0.006/tok
MAX_TOKS = 100
TEMPERATURE = 1.0
TOP_P = 1
FREQUENCY_PENALTY = 1.0
PRESENCE_PENALTY = 0.2

AI_AUTHOR = "<AI>"
TG_AUTHOR = "<TG>"
INDICATOR = "<GPT3>"
LAST_MESSAGES = deque()


def add_message(message: str, author: str):
    if len(LAST_MESSAGES) == MAX_SAVED_MESSAGES:
        LAST_MESSAGES.rotate(-1)
        LAST_MESSAGES[-1] = {"text": message, "author": author}
    else:
        LAST_MESSAGES.append({"text": message, "author": author})


def format_messages(author_to_prompt: str):
    output = ""
    for message in LAST_MESSAGES:
        output += f"{message['author']}: {message['text']}\n"
    output += f"{author_to_prompt}: "
    return output


def ask():
    convo = format_messages(AI_AUTHOR)
    response = completion.create(
        prompt=convo,
        engine=ENGINE,
        stop=[TG_AUTHOR, AI_AUTHOR],
        temperature=TEMPERATURE,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        best_of=1,
        max_tokens=MAX_TOKS,
    )
    answer = response.choices[0].text.strip()
    return answer


@client.on(events.NewMessage)
async def new_message_handler(event):
    # TODO: breaks sometimes: groups, broadcasts, permissions
    # if event.message.peer_id.user_id == MY_USER_ID:
    #     return

    from_other = event.message.message
    print(f"{TG_AUTHOR}: {from_other}")  # TODO: logging
    add_message(from_other, TG_AUTHOR)

    from_ai = ""
    while from_ai == "":
        from_ai = ask()
    print(f"{AI_AUTHOR}: {from_ai}")
    add_message(from_ai, AI_AUTHOR)

    if UWU:
        from_ai = uwuify.uwu(from_ai)
    await event.reply(f"{INDICATOR}: {from_ai}")


async def client_setup():
    await client.connect()
    await client.start()
    await client.run_until_disconnected()
    print("oops")
    return client


def main():
    for message in PRESET_PROMPT:
        add_message(message, AI_AUTHOR)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(client_setup())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A cool autoresponder for connecting services to GPT3"
    )
    parser.add_argument(
        "mode",
        choices=["telegram", "local"],
        help="Select where to respond to the user (TODO)",
    )
    parser.add_argument(
        "--uwu",
        dest="uwu",
        action="store_true",
        default=False,
        help="uwuify all responses",
    )

    ARGV = parser.parse_args()

    if ARGV.uwu:
        if UWU_LIB_AVAIL:
            UWU = True
        else:
            exit("--uwu set but uwuify is not importable!")

    main()
