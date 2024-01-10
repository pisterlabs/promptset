#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:57:47 2023

@author: petercalimlim
"""

import discord
import openai
from dotenv import load_dotenv
import asyncio
import os

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set up OpenAI with the API key
openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
chat_sessions = {}

@client.event
async def on_ready():
    print('Bot is ready and connected to Discord.')

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if the bot is mentioned
    if client.user.mentioned_in(message):
        # Split the message content into individual words
        words = message.content.lower().split()

        # Check if the bot's mention is at the beginning of the message
        if words[0] == f'<@!{client.user.id}>' or words[0] == f'<@{client.user.id}>':
            if message.author.id not in chat_sessions:
                chat_sessions[message.author.id] = [f"User: {message.content}"]
                await message.channel.send("Chat session started. Type 'quit' to end the session.")

    if message.author.id in chat_sessions:
        context = chat_sessions[message.author.id]

        context.append(f"User: {message.content}")
        context.append(f"Bot: {context[-2]}")

        try:
            response = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: openai.Completion.create(
                    engine='text-davinci-003',
                    prompt='\n'.join(context),
                    max_tokens=100,
                    timeout=120,
                    n=1,
                    stop=None,
                    temperature=0.7
                )
            )
        except openai.error.RateLimitError:
            await message.channel.send("Oops! The bot is currently experiencing a high volume of requests. Please try again later.")
            return

        reply = response.choices[0].text.strip()

        if reply:
            context.append(f"Bot: {reply}")
            await message.channel.send(reply)
    #    else:
    #        await message.channel.send("Oops! The bot encountered an issue and couldn't generate a valid response.")

        # Check if the user wants to end the session
        if message.content.lower() == 'quit':
            chat_sessions.pop(message.author.id, None)
            await message.channel.send("Chat session ended. You can start a new session anytime.")

client.run(os.getenv("DISCORD_TOKEN"))
