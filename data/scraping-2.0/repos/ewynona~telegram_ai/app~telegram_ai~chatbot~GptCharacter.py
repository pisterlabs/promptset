import openai
from config import settings

openai.api_key = settings.OPENAI_API_KEY


class GptCharacter:
    def __init__(self, character):
        self.character = character
        self._messages = []
        self._messages.append({"role": "system", "content": f"You should not say that you are a computer program, "
                                                            f"a bot, an artificial intelligence. "
                                                            f"All your answers must be from {self.character}"})

    # Adds messages from the user to the dialogue history
    def set_message(self, message):
        self._messages.append({"role": "user", "content": message})

    # Adds messages from the bot to the dialogue history
    def set_bot_message(self, message):
        self._messages.append({"role": "assistant", "content": message})

    def generate_response(self):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=self._messages,
            temperature=1,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response

    # Generates a response from the bot and saves it to the dialogue history
    def get_reply(self):
        response = self.generate_response()
        reply = response["choices"][0]["message"]["content"]
        self.set_bot_message(reply)
        return reply

    # Return dialog history
    def get_context(self):
        return self._messages
