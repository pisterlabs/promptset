import discord
from openai import OpenAI
import requests
from io import BytesIO

class MyClient(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

    async def on_message(self, message):
        if message.author == self.user:
            return

        if message.content.startswith('!generateimage'):
            prompt = message.content[len('!generateimage'):].strip()

            if not prompt:
                await message.channel.send('Please provide a prompt for the image.')
                return

            await message.channel.send('Generating image...')

            try:
                image_url = await generate_image(prompt)
                if image_url:
                    async with message.channel.typing():
                        image_data = requests.get(image_url).content  
                        image_file = discord.File(BytesIO(image_data), filename="generated_image.png")
                        await message.channel.send(file=image_file)
            except Exception as e:
                print(e)
                await message.channel.send('An error occurred while generating the image. ' + str(e))

async def generate_image(prompt):
    
    client = OpenAI(api_key='OPENAI-API-KEY')

    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    if response.data and len(response.data) > 0:
        image_url = response.data[0].url
        return image_url      
    else:
        return None

intents = discord.Intents.default()
intents.message_content = True

client = MyClient(intents=intents)
TOKEN = 'DISCORD-BOT-TOKEN'
client.run(TOKEN)
