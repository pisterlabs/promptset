""""
Copyright © Krypton 2019-2022 - https://github.com/kkrypt0nn (https://krypton.ninja)
Description:
🐍 A simple template to start to code your own and personalized discord bot in Python programming language.

Version: 5.4.1
"""

from discord import app_commands
from discord.ext import commands
from discord.ext.commands import Context
from enum import Enum

import openai

class Sizes(Enum):
    small = "256"
    medium = "512"
    large = "1024"

class ImageAI(commands.Cog, name="imageai"):
    def __init__(self, bot):
        self.bot = bot

    @commands.hybrid_command(
        name="imageai",
        description="Generate an openAI image completion",
    )
    @app_commands.describe(
        prompt="The prompt to pass to openAI",
        size="small: 256px | medium: 512px | large: 1024 px:  Default=small")
    async def openai(self, context: Context, prompt: str , size: Sizes = Sizes.small):
        dimension = "256x256" if size.value == "256" else "512x512" if size.value == "512" else "1024x1024"
        await context.defer()
        try:
            openai.api_key = self.bot.config["openai_key"]
            response = openai.Image.create(
                prompt=prompt,
                size=dimension
            )
            await context.send(response['data'][0]['url'])
        except Exception as error:
            print(f"Failed to generate image for prompt: {prompt}\nError: {error}")
            await context.send(
                f"Failed to generate image for prompt: {prompt}\nError: {error}"
            )


async def setup(bot):
    await bot.add_cog(ImageAI(bot))
