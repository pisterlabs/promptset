import os
import dbm
import json
from datetime import datetime, timezone
from PIL import Image

from transformers import pipeline

from settings import TIME_ZONE

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()
captioner = pipeline(
    "image-to-text", model="Salesforce/blip-image-captioning-base")
dir_path = os.path.dirname(os.path.realpath(__file__))


def get_messages():
    inbox = [f for f in os.listdir(
        f"{dir_path}/data/your_activity_across_facebook/messages/inbox") if not f.startswith('.')]
    dates = {}
    for message_directory in inbox:
        message_files = [f for f in os.listdir(
            f"{dir_path}/data/your_activity_across_facebook/messages/inbox/{message_directory}") if f.startswith('message_')]
        for message_file in message_files:
            with open(f"{dir_path}/data/your_activity_across_facebook/messages/inbox/{message_directory}/{message_file}", "r") as f:
                group_chat_data = json.load(f)
                name = group_chat_data["title"]
                messages = group_chat_data["messages"]
                for message in messages:
                    message["sent_at"] = datetime.utcfromtimestamp(
                        message["timestamp_ms"] / 1000).replace(tzinfo=timezone.utc)
                    date_in_timezone = message["sent_at"].astimezone(
                        TIME_ZONE).date()
                    if date_in_timezone not in dates:
                        dates[date_in_timezone] = {}
                    if name not in dates[date_in_timezone]:
                        dates[date_in_timezone][name] = []

                    dates[date_in_timezone][name].append(message)

    activities = []
    for _, chats in sorted(dates.items(), key=lambda x: x[0]):
        for chat in chats:
            min_start = min([message["sent_at"] for message in chats[chat]])
            max_end = max([message["sent_at"] for message in chats[chat]])
            message_list = []
            for message in sorted(chats[chat], key=lambda x: x["sent_at"]):
                if "content" in message and "Reacted" not in message["content"]:
                    message_list.append(
                        f"{message['sender_name']}: {message['content']}")
                elif "photos" in message:
                    for photo in message["photos"]:
                        with dbm.open('images', 'c') as db:
                            if photo['uri'] not in db:
                                img = Image.open(
                                    f"{dir_path}/data/{photo['uri']}")
                                db[photo['uri']] = captioner(
                                    img)[0]['generated_text']
                            message_list.append(
                                f"{message['sender_name']} sent a photo that looks like {db[photo['uri']]}")

            description = "Chat Name: " + chat + "\n\n"
            description += '\n'.join(message_list)
            with dbm.open('summary', 'c') as db:
                if description not in db:
                    db[description] = client.chat.completions.create(
                        model="gpt-3.5-turbo-16k",
                        messages=[
                            {"role": "system", "content": "You are an AI used to summarize conversations on facebook messenger."},
                            {"role": "user", "content": f"Summarize this conversation:\n\n{description}"},
                        ]
                    ).choices[0].message.content
                activities.append({
                    'start': min_start,
                    'end': max_end,
                    'description': db[description],
                    'type': 'Facebook Messenger Conversation',
                })

    return activities
