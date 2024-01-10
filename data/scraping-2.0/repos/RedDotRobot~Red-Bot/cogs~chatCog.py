import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import openai

load_dotenv()
openai.api_key = os.getenv("openAPIKey")

class chatCog(commands.Cog):
	def __init__(self, bot):
		self.bot = bot

	@commands.command()
	async def chat(self, ctx, *args):
		prompt = " ".join(args)
		response = openai.Completion.create(
			engine="text-davinci-002",
			prompt=prompt,
			max_tokens=1024,
			n=1,
			stop=None
)
		await ctx.send(f"**OpenAI ChatGPT**\n>>> {response}")

async def setup(bot):
	await bot.add_cog(chatCog(bot))
