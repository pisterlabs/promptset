import os
import openai
import discord
from discord.ext import commands

# Input your OpenAI API key directly
openai.api_key = "yourapikey"

# Input your Discord bot token directly
TOKEN = "yourtoken"

# Enable all intents
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='$', intents=intents)

# Keep track of conversation histories
conversation_histories = {}

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Check if a conversation history exists for the channel
    if message.channel.id not in conversation_histories:
        conversation_histories[message.channel.id] = []

    # Update conversation history with user's message
    conversation_histories[message.channel.id].append({
        'role': 'user',
        'content': message.content,
    })

    # Pass the conversation history to GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=conversation_histories[message.channel.id],
        temperature=0.6,
        max_tokens=256
    )

    # Update conversation history with assistant's response
    conversation_histories[message.channel.id].append({
        'role': 'assistant',
        'content': response['choices'][0]['message']['content'],
    })

    # Send GPT-4's response to the same channel
    await message.channel.send(response['choices'][0]['message']['content'])

bot.run(TOKEN)
