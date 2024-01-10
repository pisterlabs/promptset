from discord.ext import commands
import openai
import os

openai.api_key = os.getenv("OPEN_AI_TOKEN")


class DallE(commands.Cog):

	def __init__(self, client):
		self.client = client

	@commands.Cog.listener()
	async def on_ready(self):
		print("dall_e.py is loaded!")

	@commands.command()
	async def image(self, ctx, *, message):
		"""Get an image from Dall-E."""
		async with ctx.channel.typing():
			response = openai.Image.create(
			    prompt=message,
			    n=1,
			    size="256x256",
			)

			await ctx.channel.send(response["data"][0]["url"])


async def setup(client):
	await client.add_cog(DallE(client))
