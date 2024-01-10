import discord
import openai
import aiohttp
import io

from discord import ui
from discord import app_commands
from discord.ext import commands
from typing import Literal

from utils import config, icons

OPENAI = config["OPENAI"]


#async def pedit(prompt, instructions, temp=0.5):
#    if temp >= 1.0:
#        temp = 1
#    elif temp == 0.0 or int(temp) <= 0:
#        temp = 0
#    headers = {
#        "Content-Type": "application/json",
#        "Authorization": f"Bearer {OPENAI}"
#    }
#    json = {
#        "model": "text-davinci-edit-001",
#        "prompt": prompt,
#        "instructions": instructions,
#        "temperature": temp
#    }
#    async with aiohttp.ClientSession() as session:
#        async with session.post(
#            "https://api.openai.com/v1/edit",
#            headers=headers,
#            json=json
#        ) as response:
#            result = await response.json()
#            print(result)
#            return result["choices"][0]["text"]

#async def pedit(prompt, instructions, temp=0.5):
#	try:
#		if temp >= 1.0:
#			temp: int = 1
#
#		elif temp == 0.0 or int(temp) <= 0:
#			temp: int = 0
#
#		response = openai.Edit.create(
#			model="text-davinci-edit-001",
#			input=prompt,
#			instruction=instructions,
#			temperature=temp
#		)
#		return response.choices[0].text
#	except Exception as e:
#		return e


async def imagen(prompt, size):
	async with aiohttp.ClientSession() as session:
		try:
			# Use the OpenAI API to generate an image
			async with session.post(
				"https://api.openai.com/v1/images/generations",
				json={
					"model": "image-alpha-001",
					"prompt": prompt,
					"size": size,
					"n": 1
				},
				headers={
					"Content-Type": "application/json",
					"Authorization": f"Bearer {OPENAI}"
				}
			) as response:
				result = await response.json()
				# Extract the generated image from the response
				image_url = result["data"][0]["url"]
				async with session.get(image_url) as response:
					image_data = await response.content.read()
					print(image_data)
					return image_data
		except Exception as e:
			print(e)

async def completion(prompt, temp):
	if temp >= 1.0:
		temp = 1

	elif temp <= 0.0:
		temp = 0

	temp = round(temp, 1)

	async with aiohttp.ClientSession() as session:
		async with session.post(
			"https://api.openai.com/v1/completions",
			json={
				"model": "text-davinci-003",
				"prompt": prompt,
				"temperature": temp,
				"max_tokens": 2048
			},
			headers={
				"Content-Type": "application/json",
				"Authorization": f"Bearer {OPENAI}"
			}
		) as response:
			result = await response.json()
			return result["choices"][0]["text"]



#class EditModal(ui.Modal, title="Message completion edit"):
#	def __init__(self, prompt, result):
#		super().__init__()
#		self.prompt = prompt
#		self.result = result
#
#	instructions = ui.TextInput(label='Instructions', placeholder="The prompt for the edit. Like a completion",
#								style=discord.TextStyle.paragraph, max_length=100, required=True)
#
#	async def on_submit(self, interaction):
#		await interaction.response.defer(thinking=True)
#		embed = discord.Embed()
#		embed.set_author(name="Text Edit")
#		api_json = await pedit(self.result, str(self.instructions))
#		print(api_json)
#		embed.description = f"**{self.prompt}**\n*{self.instructions}*{api_json}"
#		view = ComplRegen(
#			self.prompt, 0.5, interaction.user.id, api_json)
#		await interaction.followup.send(embed=embed, view=view)


class ComplRegen(ui.View):
	def __init__(self, prompt, temperature, author, result):
		super().__init__()
		self.value = None
		self.prompt = prompt
		self.temperature = temperature
		self.author = author
		self.result = result

	@ui.button(label="Regenerate", style=discord.ButtonStyle.blurple, emoji=icons.regen)
	async def complregen(self, interaction: discord.Integration, button: ui.Button):
		try:

			await interaction.response.defer(thinking=True)
			api_json = await completion(self.prompt, self.temperature)
			embed = discord.Embed()
			embed.set_author(name=f"Text Completion  ·  {self.temperature}/1")
			embed.description = f"**{self.prompt}** {api_json}"
			view = ComplRegen(self.prompt, self.temperature,
							  interaction.user.id, api_json)
			msg = await interaction.original_response()
			await interaction.followup.send(embed=embed, view=view)
		except Exception as e:
			print(e)

	#@ui.button(label="Edit", style=discord.ButtonStyle.gray, emoji="<:edit:1062784953399648437>")
	#async def compledit(self, interaction, button):
	#	try:
	#		await interaction.response.send_modal(EditModal(self.prompt, self.result))
	#	except Exception as e:
	#		print(e)

	async def interaction_check(self, interaction):
		if interaction.user.id != self.author:
			await interaction.response.send_message(
				f'This is not your menu, run </openai completion:1055108797397479475> to open your own.',
				ephemeral=True)
			return False
		return True


class Remove(discord.ui.View):
	def __init__(self):
		super().__init__()

	@ui.button(label="Remove", style=discord.ButtonStyle.red, emoji=icons.remove)
	async def remove(self, interaction, button):
		try:
			await self.orimsg.delete()
		except Exception as e:
			print(e)


class ImgRegen(discord.ui.View):
	def __init__(self, prompt, size, author):
		super().__init__()
		self.value = None
		self.prompt = prompt
		self.size = size
		self.author = author

	@ui.button(label="Regenerate", style=discord.ButtonStyle.blurple, emoji=icons.regen)
	async def imgregen(self, interaction, button):
		try:
			await interaction.response.defer(thinking=True)
			image_data = await imagen(self.prompt, self.size)
			image_stream = io.BytesIO(image_data)
			file = discord.File(image_stream, filename=f'{self.prompt}.png')

			embed = discord.Embed(
				description=f"Prompt: `{self.prompt}`, Size: `{self.size}`")
			embed.set_author(name="Text to Image")
			view = ImgRegen(self.prompt, self.size, interaction.user.id)
			mesage = await interaction.followup.send(embed=embed, file=file, view=view)
			view.img = mesage.attachments[0].url
		except Exception as e:
			print(e)

	@ui.button(label="Save", style=discord.ButtonStyle.gray, emoji="<:bookmark:1062790109491114004>")
	async def bookmark(self, interaction, button):
		try:
#			image_stream = io.BytesIO(self.img)
#			file = discord.File(image_stream, filename=f'{self.prompt}.png')
			embed = discord.Embed(description=f"Prompt: `{self.prompt}`, Size: `{self.size}`")
			embed.set_author(name="Text to Image")
			embed.set_image(url=self.img)
			view = Remove()
			orimsg = await interaction.user.send(embed=embed, view=view)
			view.orimsg = orimsg
			await interaction.response.defer()
		except Exception as e:
			print(e)

	async def interaction_check(self, interaction):
		if interaction.user.id != self.author:
			await interaction.response.send_message(
				f'This is not your menu, run </openai imagen:1055108797397479475> to open your own.',
				ephemeral=True)
			return False
		return True


class OpenAI(commands.Cog):
	def __init__(self, bot):
		super().__init__()
		self.bot = bot

	openai.api_key = OPENAI

	group = app_commands.Group(
		name="openai", description="Utilize the OpenAI's api")

	@group.command(name="imagen", description="Generate images")
	@app_commands.describe(prompt="The image prompt - no default",
						   size="The image resolution - default is 512x")
	async def imagen(self, interaction, prompt: str, size: Literal['128x128', '512x512', '1024x1024'] = "512x512"):
		await interaction.response.defer(thinking=True)

		image_data = await imagen(prompt, size)
		image_stream = io.BytesIO(image_data)

		file = discord.File(image_stream, filename=f'{prompt}.png')

		embed = discord.Embed(
			description=f"Prompt: `{prompt}`, Size: `{size}`")
		embed.set_author(name="Text to Image")
		view = ImgRegen(prompt, size, interaction.user.id)
		try:
			obj = await interaction.followup.send(embed=embed, file=file, view=view)
			view.img = obj.attachments[0].url
		except Exception as e:
			await interaction.followup.send(e)

	@group.command(name="completion", description="Text completion and code generation from a prompt")
	@app_commands.describe(
		prompt="The text generation prompt - no default",
		temperature="Sampling temperature. Higher values means the model will take more risks. default is 0.5")
	async def completion(self, interaction, prompt: str, temperature: float = 0.5):
		try:
			await interaction.response.defer(thinking=True)

			api_json = await completion(prompt, temperature)
			view = ComplRegen(prompt, temperature, interaction.user.id, result=api_json)

			embed = discord.Embed()
			embed.set_author(name=f"Text Completion  ·  {temperature}/1")
			embed.description = f"**{prompt}** {api_json}"

			await interaction.followup.send(embed=embed, view=view)
		except Exception as e:
			print(e)

	#@group.command(name="edit", description="Creates a new edit for the provided input, instruction, and parameters")
	#@app_commands.describe(prompt="The text the AI will work with",
	#					   instructions="What the AI should do with the prompt",
	#					   temperature="Sampling temperature. Higher values means the model will take more risks. "
	#								   "default is 0.5")
	#async def edit(self, interaction, prompt: str, instructions: str, temperature: float = 0.5):
	#	await interaction.response.defer(thinking=True)
#
	#	api_json = await pedit(prompt, instructions, temperature)
#
	#	embed = discord.Embed()
	#	embed.set_author(name="Text Edit")
	#	embed.description = f"**{prompt}**\n*{instructions}*\n\n{api_json}"
	#	await interaction.followup.send(embed=embed)


async def setup(bot):
	await bot.add_cog(OpenAI(bot))
