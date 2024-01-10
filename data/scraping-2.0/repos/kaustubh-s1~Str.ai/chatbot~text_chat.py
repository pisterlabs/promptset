import openai
import os
from dotenv import load_dotenv
from utils.prompt import PROMPT_FOR_GPT

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

class TextChatApp:
    """
    This is a Python class for a text chat application that uses OpenAI's GPT-3 model to generate
    responses to user input.
    
    :param user_input: The input message from the user that will be used as context for the GPT-3
    model to generate a response
    """
    def __init__(self):
        self.messages = [
            {"role": "system", "content": PROMPT_FOR_GPT}
        ]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        chatgpt_reply = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": chatgpt_reply})

        chat_transcript = ""
        for message in self.messages:
            if message['role'] != 'system':
                chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

        return chat_transcript
