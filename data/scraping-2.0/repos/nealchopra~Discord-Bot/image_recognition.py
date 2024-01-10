import openai
import requests
import json
import base64
import io
import discord

client = discord.Client()

openai.api_key = config.OPENAI_API_KEY

async def generate_image(description):
    response = openai.Image.create(
        prompt=description
    )
    img_data = response['data']
    img_bytes = base64.b64decode(img_data)
    img = io.BytesIO(img_bytes)
    return img

@client.event
async def on_message(message):
    try:
        if message.content.startswith("!image"):
            description = message.content[7:]
            image = await generate_image(description)
            await message.channel.send(file=discord.File(image, 'image.png'))
    
    except Exception as e:
        await message.channel.send("Sorry, something went wrong. Please try again later.")
        print(e)
