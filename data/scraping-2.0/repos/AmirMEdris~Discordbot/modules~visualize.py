
import os
import discord
import requests
import torch
from diffusers import StableDiffusionPipeline
from discord.ext import commands
import openai
import datetime


class Visualize(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print("Visualize cog initialized")
        self.model_id = "dreamlike-art/dreamlike-photoreal-2.0"
        self.pipe = StableDiffusionPipeline.from_pretrained(self.model_id, torch_dtype=torch.float16)
        self.pipe = self.pipe.to("cuda")

    async def create_image(self, text):
        image_path = 'generated_image.png'
        prompt = text
        image = self.pipe(prompt).images[0]

        image.save(image_path)
        return image_path

    # Listen for "visualize" command
    @commands.command(name='visualize')
    async def visualizefree(self, ctx, *, text):
        image_path = await self.create_image(text)
        with open(image_path, 'rb') as img_file:
            # Create and send Discord file
            discord_file = discord.File(fp=img_file, filename='visualized_text.png')
            await ctx.send(file=discord_file)

        # Remove the generated image file
        os.remove(image_path)