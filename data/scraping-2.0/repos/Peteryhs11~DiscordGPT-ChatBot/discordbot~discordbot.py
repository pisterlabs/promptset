#discordbot that uses the openai api to talk with the use

import discord
import os
import openai
from dotenv import load_dotenv

load_dotenv()
#grabs your openai api key and discord bot token from the .env file
openai.api_key = os.getenv('openai_api_key')
discord_bot_token = os.getenv("discord_bot_token")

#sets the discord client
client = discord.Client(intents=discord.Intents.all())

#when the bot is ready
@client.event
async def on_ready():
    print("Discord bot is up as: {0.user}".format(client))

#when the bot recieves a message
@client.event
async def on_message(message):
    #if the message is from the bot itself, ignore it
    if message.author == client.user:
        return

    #if the message starts with the command prefix
    if client.user in message.mentions:
        #get the message without the command prefix


        #send the message to the openai api
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
            
                {"role": "system", "content": "You are a helpful AI discord chatbot"},#You can change the personality of the bot by changing this line
                {"role": "user", "content": message.content}
            ]
        )

        #send the response from the openai api to the discord channel
        await message.channel.send(response.choices[0].message.content)

#run the bot
client.run("YOUR DISCORD BOT TOKEN")












