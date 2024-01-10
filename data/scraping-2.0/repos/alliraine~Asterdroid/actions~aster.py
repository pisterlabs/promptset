from signalbot import Command, Context
import openai
import os

class AsterCommand(Command):
    def describe(self) -> str:
        return "OpenAI Trigger"

    async def handle(self, c: Context):
        command = c.message.text
        print(c.message.mentions)
        if len(c.message.mentions) > 0:
            if "2219a1a4-828b-4af1-9804-ca839236d40f" in c.message.mentions[0]["uuid"]:
                await c.start_typing()
                await c.react('🤖')
                openai.api_key = os.getenv("OPEN_AI")
                response = openai.ChatCompletion.create(
                  model="gpt-3.5-turbo",
                  messages=[
                    {"role": "system", "content": "You are a chat bot called Asterbot. You are here to assisted The Constellation a queer lesbian polycule which consists of Alli, Jen, Ellie, and Sae."},
                    {"role": "user", "content": command}
                  ]
                )
                await c.send(response.choices[0].message.content)
                await c.stop_typing()
                return
