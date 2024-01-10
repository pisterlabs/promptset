from openai import OpenAI

#f = open("key.txt", "r")
#openai.api_key = f.read()  # preberemo API kljuc

client = OpenAI(api_key="")

class Conversation:
    def __init__(self):
        self.active = True  # pogovor poteka

        self.messages = []  # shranjena preteklost pogovora

    def setup(self, sys_prompt):
        # sistemsko navodilo agentu
        self.messages.append({"role": "system", "content": sys_prompt})

    def new_prompt(self, text):
        # nadaljna navodila, ki prihajajo od uporabnika
        if not self.active:
            return ""

        self.messages.append({"role": "user", "content": text})
        chat = client.chat.completions.create(
            model="gpt-3.5-turbo-1106", messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.messages.append({"role": "assistant", "content": reply})

        return reply
