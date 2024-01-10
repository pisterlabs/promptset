import os
from openai import OpenAI


class ChatBot:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("ChatBot"))

    def get_response(self, user_input):
        """ A function interact with chat GPT """
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": user_input}
            ]
        ).choices[0].message.content

        return response
