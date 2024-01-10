import asyncio
import json
import os
import pickle
from os import path

import openai
from dotenv import load_dotenv

from gptme.adapters import DiscordAdapter
from gptme.assistant import Assistant
from gptme.conversation import Message
from gptme.text_styler import TextStyler

load_dotenv(dotenv_path=".env")

openai.api_key = os.environ["OPENAI_KEY"]

with open("personality.txt", encoding="utf8") as personality_file:
    personality = personality_file.read()
    print("Loaded personality.")

with open(os.environ["EMBEDDINGS_FILE"], "rb") as embeddings_file:
    embeddings, transcript = pickle.load(embeddings_file)
    print("Loaded memories.")

with open(os.environ["STYLER_FILE"], "r", encoding="utf8") as style_file:
    style_json = json.load(style_file)
    text_styler = TextStyler()
    text_styler.style = style_json["style"]
    print("Loaded styler.")

if path.exists(".memories/conversation.json"):
    with open(".memories/conversation.json", encoding="utf8") as conversation_file:
        past_conversation = json.load(conversation_file)
        past_conversation = [
            Message(content=message["content"], role=message["role"])
            for message in past_conversation
        ]
        print("Loaded past conversation.")
else:
    past_conversation = []

assistant = Assistant(
    embeddings=embeddings,
    memories=transcript,
    personality=personality,
    name=os.environ["ASSISTANT_NAME"],
    text_styler=text_styler,
    messages=past_conversation,
)

print("Prepared assistant.")

discord = DiscordAdapter(
    assistant=assistant, channel_id=int(os.environ["DISCORD_CHANNEL"])
)

print("Prepared adapter.")

discord.run(os.environ["DISCORD_TOKEN"])
