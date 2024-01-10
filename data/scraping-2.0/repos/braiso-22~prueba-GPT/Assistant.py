from typing import List
import openai
import getpass

from Message import Message
import dotenv


def initialize_openai():
    openai.organization = "org-90nlZdVZc1Zg3BvFf3xIVfmD"
    openai.api_key = dotenv.get_key(".env", "OPENAI_API_KEY")
    return openai


class Assistant:
    def __init__(self, name: str, owner: str = getpass.getuser(), messages: List[Message] = []):
        self.name: str = name
        self.owner: str = owner
        self.model = initialize_openai()
        self.messages: List[Message] = messages

    def add_message(self, message: Message):
        self.messages.append(message)

    def send_conversation(self):
        messages = [message.to_dict() for message in self.messages]
        try:
            chat_completion = self.model.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
            )
        except Exception as e:
            return e
        return chat_completion

    def create_completion(self, prompt: str):
        completion = self.model.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=4000,
        )
        return completion

    def initial_configuration(self, configuration: str = ""):
        if configuration == "":
            configuration = f"Eres una IA llamada {self.name} y tu objetivo  es responder a las preguntas del usuario."
        if self.owner:
            configuration += f""" El nombre de tu usario es {self.owner}."""
        clean_input = configuration.replace("\n", "").strip()
        system_message = Message(
            sender=Message.SYSTEM,
            content=clean_input
        )
        self.add_message(system_message)

    def generar_imagen(self, prompt: str):
        response = self.model.Image.create(
            prompt=prompt,
            n=1,
            size="512x512"
        )
        image_url = response['data'][0]['url']
        return image_url
