import os
import discord
from discord.ext import commands
import gym
import numpy as np
from sklearn.linear_model import LinearRegression
import openai

# Set up your OpenAI API key
openai.api_key = os.getenv('')

# Define intents
intents = discord.Intents.default()
intents.typing = False
intents.presences = False

# Create a bot instance with a command prefix
bot = commands.Bot(command_prefix='Ask ', intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')

# Add your commands here

# Run your bot with the token
token = os.getenv('')

bot.run('')
