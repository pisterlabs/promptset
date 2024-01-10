import os
import json
import jsonlines
import csv
from dotenv import load_dotenv
from openai import OpenAI
from io import StringIO

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class EmotionExtractionHandler:
    def __init__(self):
        self.fine_tuned_model = "ft:gpt-3.5-turbo-0613:universitat-polit-cnica-de-catalunya::8bOqoLfh"

    def emotion_extraction(self, text):
        try:
            results = []
            for message in text:
                completion = client.chat.completions.create(
                    model=self.fine_tuned_model,
                    messages=[
                        {"role": "system", "content": "Can you tell me what emotion is expressing the message above? It can be one of the following: happiness, sadness, anger, fear, surprise, disgust or not-relevant (if it doesn't express any relevant emotion)."},
                        {"role": "user", "content": message['text']}
                    ]
                )
                if completion is None:
                    return "An error occurred"
                results.append({'text': message, 'emotion': completion.choices[0].message.content})

            return results

        except Exception as e:
            print("\033[91m\u2718 " + f"An error occurred: {e}")
            return None