import json
from datetime import datetime

import openai


class GPT35Turbo:
    def __init__(self, api_key, model="gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        self.messages = []
        self.filename = self.get_filename()

    def add_system_message(self, message):
        self.messages.append({"role": "system", "content": message})

    def add_user_message(self, message):
        self.messages.append({"role": "user", "content": message})

    def generate_response(self):
        self.save_conversation()
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.2
            # stream=True
        )
        role = response["choices"][0]["message"]["role"]
        self.messages.append({"role": role, "content": response["choices"][0]["message"]["content"]})
        return response["choices"][0]["message"]["content"]

    def clear_messages(self):
        self.messages = []

    def summarize_messages(self):
        pass

    def get_filename(self):
        now = datetime.now()
        return now.strftime("%d-%m-%Y-%H-%M-%S")

    def save_conversation(self):
        path = "Conversations\\" + self.filename + ".txt"
        with open(path, "w") as f:
            for message in self.messages:
                json.dump(message, f)
                f.write('\n')