import os
import openai
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load the API keys from the .env file
load_dotenv()
BARNEY_TOKEN = os.getenv('BARNEY_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize the OpenAI API
openai.api_key = OPENAI_API_KEY

# Set up the Discord bot
intents = discord.Intents.default()
intents.typing = False
intents.presences = False
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

async def get_gpt_response(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Pretend you are the character Barney Gumble from the Simpsons. Please be aloof in presentation, but surprisingly insightful in terms of content. Please limit your responses to 250 tokens."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        n=1,
        temperature=0.8,
    )

    return response.choices[0].message['content']

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the message starts with an exclamation mark
    if message.content.startswith('!'):
        prompt = message.content[1:]  # Remove the exclamation mark from the message
        response = await get_gpt_response(prompt)
        await message.channel.send(response)

# Run the bot
if __name__ == "__main__":
    bot.run(BARNEY_TOKEN)
