import discord
import requests
import json
import io
import openai

api_key = "" # Input OpenAI Key here in quotes as a string
discord_token = "DISCORD TOKEN HERE"  # Input Discord Token here
model_name = "ENGINE MODEL NAME HERE"  # Input Engine Model Name here e.g. "image-alpha-001"

# Discord bot setup
intents = discord.Intents().all()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

@client.event
async def on_message(message):
    if message.content == 'bots.shutdown': 
        await message.channel.send('Shutting down...')
        await client.close()
    if message.content.startswith('%'):  # Start discord message with % to generate image based on following text
        prompt = message.content[1:]
        try:
            response = openai.Image.create(  # See API documentation for further parameters
                model=model_name,
                prompt=prompt,
                num_images=1,
                size="512x512",
                response_format="url",
                api_key=api_key
            )
            image_url = response['data'][0]['url']
            file = discord.File(io.BytesIO(requests.get(image_url).content), filename="image.jpg")
            await message.channel.send(file=file)
        except (requests.exceptions.RequestException, json.JSONDecodeError, openai.error.OpenAIError) as e:
            print(e)
            await message.channel.send("Failed to generate image. Please try again later.")

client.run(discord_token)  
