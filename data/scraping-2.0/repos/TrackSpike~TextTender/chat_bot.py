import openai
import json
import os
from gtp_message_context import GtpMessageContext


class ChatBot:
    def __init__(self, target, system_message):
        self.target = target
        self.system_message = system_message
        self.config = json.load(open("config.json"))

    def send_message(self, message):
        print(f"Sending message to {self.target} with message: {message}")
        self.context.add_assistant_message(message)
        if not self.config["TEST_MODE"]:
            os.system(f'osascript sendMessage.applescript {self.target} "{message}"')

    def generate_response(self):
        print(self.context)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.context.to_json(),
        )
        return response.choices[0].message.content.replace("'", "").replace('"', "")

    def build_context(self, depth, messages):
        self.context = GtpMessageContext(
            system_message=self.system_message, maxlen=depth
        )
        # Docs state that the messages are sorted by order but wasn't the case for me
        # Sorting them manually here works.
        messages = [x for x in messages if x.user_id == self.target]
        messages.sort(key=lambda x: x.date)

        for message in messages[::-1]:
            if message.is_from_me:
                self.context.add_assistant_message(message.message, prepend=True)
            else:
                self.context.add_user_message(message.message, prepend=True)

            if self.context.is_full:
                print("Context built: \n", self.context)
                return
