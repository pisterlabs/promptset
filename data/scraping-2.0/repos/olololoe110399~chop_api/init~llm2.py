import json
import os

from openai import Client


class Recommender:
    def __init__(self):
        self.client = Client()
        self.max_length = 5000
        self.messages = []
        with open(os.path.join(os.getcwd(), 'resources/system_message'), 'r') as file:
            self.system_message = file.read()

        self.messages.append(
            {
                "role": "system",
                "content": self.system_message
            }
        )

    def check_message_length(self):
        messages_length = 0
        for message in self.messages:
            messages_length += len(message['content'].split(' '))
        if messages_length > self.max_length:
            while messages_length > self.max_length:
                self.messages.pop(0)
                messages_length -= len(self.messages[1]['content'].split())
                self.messages.pop(1)

    def recommend(self, role, input):
        self.messages.append({"role": role, "content": input})
        try:
            response = self.client.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                response_format={"type": "json_object"},
                stream=True,
                messages=self.messages,
            )
        except Exception as e:
            raise e
        responses = ''
        first_chunk = True
        user_response_printing = False

        for chunk in response:
            if chunk.choices[0].finish_reason is not None:
                break
            if chunk.choices[0].delta.content:
                text_chunk = chunk.choices[0].delta.content
                responses += str(text_chunk)

                try:
                    if 'your response to user": "' in responses:
                        user_response_printing = True
                        if first_chunk:
                            text_chunk = text_chunk.lstrip(' "')
                            first_chunk = False
                        if '"' or '}' in text_chunk:
                            if text_chunk.count('"') % 2 != 0:
                                text_chunk = text_chunk.rstrip('"\n')
                            text_chunk = text_chunk.rstrip('}')
                except json.JSONDecodeError as e:
                    pass
                print(text_chunk, end="", flush=True)
        self.messages.append({"role": "assistant", "content": responses})
        self.check_message_length()
        return json.loads(responses)


