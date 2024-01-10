import os
import discord
from openai import OpenAI, ChatCompletion

# Configura tu clave API de OpenAI
openai = OpenAI(os.getenv("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"))

# Configura tu bot de Discord
class MyBot(discord.Client):
    async def on_ready(self):
        print(f'We have logged in as {self.user}')

    async def on_message(self, message):
        if message.author == self.user:
            return

        # Configura tu modelo de OpenAI
        model = "gpt-3.5-turbo"
        messages = [{"role": "system", "content": "You are chatting with an AI assistant."},
                    {"role": "user", "content": message.content}]

        # Realiza una llamada a la API de OpenAI
        response = ChatCompletion.create(model=model, messages=messages)
        bot_message = response['choices'][0]['message']['content']

        # Envía la respuesta del bot al canal de Discord
        await message.channel.send(bot_message)

# Inicia tu bot de Discord
client = MyBot()
client.run(os.getenv("MTE3NTYxNDQyNjUzNjIxODcyNg.GLzbSt.34TUTgLEmxjXBGCvuRVNAXHVYzea58LPWb1VRs"))