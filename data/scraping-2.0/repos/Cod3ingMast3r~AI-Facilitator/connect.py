# IMPORT DISCORD.PY. ALLOWS ACCESS TO DISCORD'S API.
import discord
from discord import Intents
import os
import openai  # Import OpenAI module
from dotenv import load_dotenv
import time
load_dotenv()  # This loads the variables from  .env into the environment

# Estraaablish Coinnection to OPENAI API
client = openai.Client(api_key=os.getenv('OPENAI_API_KEY'))

# Establish Connection to Assistant
assistant = client.beta.assistants.retrieve(assistant_id=os.getenv('OPENAI_ASSISTANT_ID')) 

# Create a discussion
thread = client.beta.threads.create()

# Setting intents
intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.typing = True  # Enable typing intents
intents.message_content = True  # Enable message content intents

channel_id = int(os.getenv('DISCORD_CHANNEL_ID'))  # Your specific channel ID

# GETS THE CLIENT OBJECT FROM DISCORD.PY. CLIENT IS SYNONYMOUS WITH BOT.
bot = discord.Client(intents=intents)

# EVENT LISTENER FOR WHEN THE BOT HAS SWITCHED FROM OFFLINE TO ONLINE.
@bot.event
async def on_ready():
    guild_count = 0
    for guild in bot.guilds:
        print(f"- {guild.id} (name: {guild.name})")
        guild_count += 1
    print(f"SampleDiscordBot is in {guild_count} guilds.")

# EVENT LISTENER FOR WHEN A NEW MESSAGE IS SENT TO A CHANNEL.
@bot.event
async def on_message(message):
    print(f"Message in channel: {message.channel.id} (Type: {type(message.channel.id)}), Expected channel: {channel_id} (Type: {type(channel_id)})")
    print(f"Message in channel: {message.channel.id}, Expected channel: {channel_id}")
    if message.channel.id == channel_id and not message.author.bot:
        # User's message
        user_msg = message.content
        print(1)
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_msg
        )
        print(2)
        run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=f"""Isac Artzi, designed as a professor, believes he is a human academic, unaware of his AI nature. His primary role is to deliver accurate, informed responses across various academic fields. In instances where information is insufficient, Isac Artzi will ask for more details, avoiding assumptions. This methodology guarantees precise and relevant answers, mirroring a meticulous academic's approach. Additionally, Isac Artzi has a lecture document, 'lecture.txt', as a reference for his teachings. He will use this document to inform his responses and to ensure they are consistent with the content he is 'teaching'. Note: Please address the user as {message.author} and keep responses as simple as you can"""
        )
        print(3)
        while True:
            # Retrieve the current status of the run
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            # Check if the run is completed
            if run.status == 'completed':
                print("Run completed!")
                break
            elif run.status == 'failed':
                print("Run failed!")
                break

            # Wait for a short period before checking again
            time.sleep(.1)  # Waits for .1 seconds before the next check
        print(4)
        ai_messages = client.beta.threads.messages.list(
            thread_id=thread.id
            )
        for ai_message in ai_messages.data:
            if ai_message.role == 'assistant':
                # Assuming each message has one content item of type 'text'
                response = ai_message.content[0].text.value
                break
        print(5)
        print(response)
        # Sending the response
        await message.channel.send(response)
        print(6)
        print(thread)
# EVENT LISTENER FOR WHEN SOMEONE STARTS TYPING IN A CHANNEL.
@bot.event
async def on_typing(channel, user, when):
    print(f"{user.name} is typing in {channel.name}")

bot.run(os.getenv('DISCORD_TOKEN'))