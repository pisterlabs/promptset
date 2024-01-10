# consumers.py

import json
import re
import openai
import os
from channels.generic.websocket import AsyncWebsocketConsumer
from collections import defaultdict
import httpx
from decouple import config

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = config("OPENAI_API_KEY")


class SentimentConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.connected = False

    async def connect(self):
        print("Connected")
        self.connected = True
        await self.accept()

    async def disconnect(self, close_code):
        print("Disconnected")
        self.connected = False

    async def receive(self, text_data):
        data_json = json.loads(text_data)
        file_path = data_json["file_path"]
        async for avg_sentiment in self.analyze_subtitle2(file_path):
            await self.send(json.dumps({"avg_sentiment": avg_sentiment}))
        await self.send(json.dumps({"type": "end"}))

    async def analyze_subtitle2(self, file_path):
        with open(file_path, "r") as file:
            text = file.read()
            blocks = re.findall(
                r"(\d{2}:\d{2}:\d{2},\d+) --> (\d{2}:\d{2}:\d{2},\d+)\n(.*?)\n\n",
                text,
                re.DOTALL,
            )

            text_by_minute = defaultdict(str)

            for start, end, text in blocks:
                hour, minute, second, _ = map(int, re.split("[:,]", start))
                total_minutes = hour * 60 + minute
                clean_text = re.sub(r"\W+", " ", text).replace("\n", " ").strip()
                text_by_minute[total_minutes] += " " + clean_text

            sentiment_sum = 0
            count = 0

            for idx, (minute, text) in enumerate(sorted(text_by_minute.items())):
                if not self.connected:
                    break
                response = await self.get_sentiment(text, minute)
                if "choices" in response and response["choices"]:
                    response_content = response["choices"][0]["message"][
                        "content"
                    ].strip()
                    parts = response_content.split(",")
                    if len(parts) >= 2:
                        sentiment_score = parts[0].strip()
                        reason = ",".join(parts[1:]).strip()
                        print(f"Score: {sentiment_score}, Reason: {reason}")
                        yield (
                            sentiment_score,
                            reason,
                        )

                    else:
                        print(
                            "Invalid response format: Unable to extract sentiment and reason."
                        )
                else:
                    print("Invalid response format: Missing 'choices' in response.")

    async def get_sentiment(self, text, minute):
        if not self.connected:
            return {"choices": [{"message": {"content": "5"}}]}
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }

        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant who speaks like a pirate and gives sarcastic responses.",
                },
                {
                    "role": "user",
                    "content": f'Sentiment analysis of the text from minute {minute} to {minute}: "{text}". Give a sentiment score between 0 and 10 and a short, sarcastic reason for the score, speakin like a pirate, arr! Keep it very short!. Use the following format: (number), (reason)',
                },
            ],
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(OPENAI_API_URL, headers=headers, json=data)

        if response.status_code != 200:
            print(f"Error: {response.status_code}, {response.text}")
            return {"choices": [{"message": {"content": "5"}}]}

        return response.json()
