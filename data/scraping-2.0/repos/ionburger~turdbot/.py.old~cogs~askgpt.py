import discord
from discord.ext import bridge, commands
import openai

class Askgpt(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
    
    @bridge.bridge_command()
    async def askgpt(self, ctx, *, message):

