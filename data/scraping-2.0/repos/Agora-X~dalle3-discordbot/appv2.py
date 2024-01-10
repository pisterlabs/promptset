import os

import boto3
import discord
import openai
import requests
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# Fetch keys from environment variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
# S3_BUCKET_NAME=os.getenv("S3_BUCKET_NAME")


# # AWS S3 Configuration
# s3 = boto3.client(
#     "s3",
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY
# )


openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.messages = True
intents.guilds = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name}")


# Upoad to s3
# async def upload_to_s3(file_name, data):
#     try:
            
#         s3.put_object(
#             Bucket=S3_BUCKET_NAME,
#             Key=file_name,
#             Body=data
#         )
#         return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
#     except Exception as error:
#         print(f"An error occurred: {error}")
#         # return None
    

@bot.command()
async def generate(ctx, *, prompt: str = None):
    """Generates a response or image based on the provided prompt or image"""
    if prompt:
        # Text prompt provided, generate image with DALL·E 3
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url

        # if image_url:
        #     response = requests.get(image_url)
        #     if response.status_code == 200:
        #         s3_url = await upload_to_s3("image.png", response.content)
        #         if s3_url:
        #             await ctx.send(f"Image generated and saved to s3: {s3_url}")
        #         else:
        #             await ctx.send("An error occurred while uploading to S3")

        await ctx.send(image_url)
    elif ctx.message.attachments:
        # Image attached, use GPT-4 with Vision
        image_url = ctx.message.attachments[0].url
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": {"image": image_url}}
            ],
            max_tokens=300
        )
        await ctx.send(response.choices[0].message['content'])
    else:
        await ctx.send("Please provide a text prompt or an image.")

@generate.error
async def generate_error(ctx, error):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("You must provide a prompt or an image!")
    else:
        await ctx.send(f"An error occurred: {error}")

bot.run(DISCORD_TOKEN)
