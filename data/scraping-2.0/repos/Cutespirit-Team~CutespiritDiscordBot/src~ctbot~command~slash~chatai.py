import openai
import os
import discord
from discord.ext import commands
from ..utils import cog_slash_managed
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def prompt(input_str):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "user", "content": input_str}
        ]
    )
    res = response.choices[0].message.content
    return res



class SlashChatai(commands.Cog):
    def __init__(self, bot: discord.Client):
        self.bot = bot
    @cog_slash_managed(description='OpenAI GPT-3.5')
    async def chatai(self, ctx, msg):
        await ctx.defer()
        await ctx.send(prompt(msg))
