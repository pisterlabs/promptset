import discord
from discord.ext import commands
import openai

from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")


class Imagine(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @discord.slash_command(name="imagine", description="AI generate a picture")
    async def imagine(
        self, ctx, user_prompt: discord.Option(discord.SlashCommandOptionType.string)
    ):
        await ctx.defer()
        await ctx.respond("Loading...")
        response = openai.Image.create(prompt=user_prompt, n=1, size="1024x1024")
        image_url = response["data"][0]["url"]
        embed = discord.Embed(
            title=user_prompt,
            color=discord.Colour.blurple(),  # Pycord provides a class with default colors you can choose from
        )
        embed.set_image(url=image_url)
        await ctx.edit(embed=embed)


def setup(bot):
    bot.add_cog(Imagine(bot))
