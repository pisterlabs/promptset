import os
from discord import discord, Intents
from dotenv import load_dotenv
import openai

# get the grand-parent directory of the current file
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(script_dir, '.env')

# load the environment variables from the .env file
load_dotenv(dotenv_path=env_path)

TOKEN = os.getenv('DISCORD_TOKEN')
intents = Intents.default()
intents.message_content = True
discord_client = discord.Client(intents=intents)

@discord_client.event
async def on_ready():
    print(f'{discord_client.user} has connected to Discord!')

@discord_client.event
async def on_message(message):
    if message.author == discord_client.user:
        return

    print("on message: ", message.content)

    if message.content.startswith('!generate'):
        prompt = message.content[9:].strip()
        response = openai.Completion.create(
            engine="davinci-codex",
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5,
        )
        reply = response.choices[0].text.strip()
        await message.channel.send(reply)
    elif "!turbo" in message.content:
        prompt = message.content.strip()
        messages = [
            {"role": "system", "content": "you are a chat bot."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            #prompt=prompt,
            # temperature=0.7,
            # max_tokens=256,
            # top_p=1,
            # frequency_penalty=0,
            # presence_penalty=0
            messages=messages
        )
        print(response)
        reply = response.choices[0].message.content.strip()
        await message.channel.send(reply)

discord_client.run(TOKEN)