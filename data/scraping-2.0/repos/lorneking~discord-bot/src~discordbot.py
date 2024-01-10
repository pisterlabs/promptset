# ChatGPT Discord Bot
# 2023 Lorne King

import discord
from discord.ext import commands
import openai
from openai import OpenAI
import os

# Initialize GPT-4 API client
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Create an instance of the intents class
intents = discord.Intents.default()
#intents = discord.Intents.none()
intents.message_content = True
intents.messages = True

# Initialize Discord Bot
bot = commands.Bot(command_prefix='/', intents=intents)

supported_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-1106-preview"] # List of supported models

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.command()
async def ask(ctx, model="gpt-4", *, question):
    if model not in supported_models:
        await ctx.send("Invalid model selected. Please choose from: " + ", ".join(supported_models))
        return
    
    try:
        # Call GPT API with the specified model
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": question}
            ]
        )
        await ctx.send(completion.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
        await ctx.send("An error occurred while processing your request.")  

# Run the bot
bot.run(os.getenv("DISCORD_BOT_TOKEN"))



