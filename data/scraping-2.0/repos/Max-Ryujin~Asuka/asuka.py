import discord
import openai
import os
import random

intents = discord.Intents.default()


# discord bot 
client = discord.Client(intents=intents)

# openai api key
openai.api_key = os.getenv("OPENAI_API_KEY")

# discord bot token
token = os.getenv("DISCORD_TOKEN")

# load prompt from file systemprompt.txt
systemprompt = open("systemprompt", "r").read()

# bot main function
@client.event
async def on_message(message):

    # we do not want the bot to reply to itself
    if message.author == client.user:
        return
    
    #every message that is addressed at the bot with @
    if client.user.mentioned_in(message):
        prompt = ""
        #get a list of the last few messages in the channel in reverse 
        async for msg in message.channel.history(limit=3):
            #add the message and sender to the prompt
            prompt = f"{msg.content} \n" + prompt
            prompt = prompt.replace(f"<@!{msg.author.id}>", msg.author.name)

        prompt = prompt.replace(f"<@!{client.user.id}>", "Asuka")
        
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": prompt}
            ]
        )

        await message.channel.send(response['choices'][0]['message']['content']) 
    
    #50% chance that the bot ansers if the messange ends with a question mark
    elif message.content.endswith("?") and random.random() > 0.5:
        prompt = message.content
        #get a list of the last few messages in the channel in reverse 
        async for msg in message.channel.history(limit=3):
            #add the message and sender to the prompt
            prompt = f"{msg.content} \n" + prompt
            prompt = prompt.replace(f"<@!{msg.author.id}>", msg.author.name)

        prompt = prompt.replace(f"<@!{client.user.id}>", "Asuka")

        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": systemprompt},
                {"role": "user", "content": prompt}
            ]
        )

        await message.channel.send(response['choices'][0]['message']['content'])
        return



#bot start 
client.run(os.getenv("DISCORD_TOKEN"))