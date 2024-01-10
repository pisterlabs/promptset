import os
import io
import base64
import string
import random
from dotenv import load_dotenv
import discord
from discord.ext import commands
import aiohttp
import openai
# referencing https://github.com/Kilvoctu/aiyabot

## setup

load_dotenv()
intents = discord.Intents.all()
client = commands.Bot(intents=intents, command_prefix='$')
openai.api_key = os.getenv('OPENAI')
SD_URL: str = 'http://localhost:7860'


class Game:
	def __init__(self, channel):
		self.channel = channel
		# keys are mentions
		self.player_prompts: dict[str, str] = {}
		self.player_images_b64: dict[str, str] = {}
		self.currentIndex: int = 0
		
		self.show_on_finish = False
	
	async def show_all_images(self):
		await self.channel.send('Showing all images')
		for player, image_url in self.player_images_b64.items():
			member = next((v for v in self.channel.members if v.mention == player), None)
			prompt = self.player_prompts[player]
			embed=discord.Embed(title="", url=image_url, description="", color=0x1CEEEE)
			embed.set_thumbnail(url=image_url)
			await self.channel.send(f'{member.display_name}: {prompt}', embed=embed)

	async def show_next_image(self):
		await self.channel.send('Showing next image')
		image_url = self.player_images_b64[list(self.player_images_b64.keys())[self.currentIndex]]

		player = list(self.player_images_b64.keys())[self.currentIndex]
		self.currentIndex += 1
		prompt = self.player_prompts[player]
		embed=discord.Embed(title="", url=image_url, description="", color=0x1CEEEE)
		embed.set_thumbnail(url=image_url)
		guessName = "Guess who created this"
		await self.channel.send(f'{guessName}', embed=embed)

# game code -> game
current_games: dict[str, Game] = {}
# channel id -> game code
channel_to_code: dict[str, str] = {}


CODE_CHARS = string.ascii_uppercase.translate({ord(i): None for i in 'IO'})
CODE_LEN: int = 5
def generate_code() -> str:
	# todo: word filter
	while True:
		code = ''.join(random.choices(CODE_CHARS, k=5))
		if code not in current_games:
			return code

async def generate_image(prompt: str) -> str:
	print(f'Generating "{prompt}"')
	response = openai.Image.create(
    prompt=prompt,
    n=1,
    size="1024x1024"
    )
	image_url = response['data'][0]['url']

	return image_url


## events and commands

@client.event
async def on_ready():
	print(f'Logged in as {client.user}')

@client.event
async def on_message(msg):
	# accept prompts from DMs
	if msg.channel.type == discord.ChannelType.private:
		msg_parts = msg.content.split(' ')
		if len(msg_parts) >= 2 and len(msg_parts[0]) == CODE_LEN and all(c.isupper() for c in msg_parts[0]):
			# invalid code
			if msg_parts[0] not in current_games:
				await msg.channel.send(f'Game "{msg_parts[0]}" does not exist')
				return
			
			# set player's prompt
			game: Game = current_games[msg_parts[0]]
			if msg.author.mention in game.player_prompts:
				await msg.channel.send(f'Prompt already set: "{game.player_prompts[msg.author.mention]}"')
				return
			prompt = ' '.join(msg_parts[1:])
			game.player_prompts[msg.author.mention] = prompt
			
			# notify that player is ready
			member = next((v for v in game.channel.members
				if v.mention == msg.author.mention), None)
			if member is None:
				await game.channel.send("Someone tried to join, but they don't have access to this channel")
			else:
				await game.channel.send(f'{member.display_name} is ready')

			# generate image
			game.player_images_b64[msg.author.mention] = await generate_image(prompt)
			# if $show was sent
			if game.show_on_finish and game.player_prompts.keys() == game.player_images_b64.keys():
				await game.show_all_images()
				game.show_on_finish = False
			
			return
	
	# otherwise fall through to commands
	await client.process_commands(msg)


@client.command()
async def generate(ctx):
	prompt = ctx.message.content[len('$generate '):].strip()
	image_base64 = await generate_image(prompt)
	file = discord.File(io.BytesIO(base64.b64decode(image_base64)), filename='image.jpg')
	await ctx.send(f'Generated "{prompt}"', file=file)

@client.command()
async def newgame(ctx):
	# clean up previous game
	prev_code = channel_to_code.get(ctx.channel.id)
	prev_game = current_games.get(prev_code)
	if prev_game is not None:
		del channel_to_code[ctx.channel.id]
		del current_games[prev_code]
	
	# start new game
	code: str = generate_code()
	current_games[code] = Game(ctx.channel)
	channel_to_code[ctx.channel.id] = code
	await ctx.send(f'Starting game {code}. Join by DMing a prompt to me like this:\n'
		+ f'{code} [prompt]')

@client.command()
async def showall(ctx):
	code: str = channel_to_code.get(ctx.channel.id)
	game: Game = current_games.get(code)
	if game is None:
		await ctx.send('Start a new game with "$newgame" first')
		return
	
	if game.player_prompts.keys() != game.player_images_b64.keys():
		await ctx.send('Still generating images...')
		game.show_on_finish = True
		return
	
	await game.show_all_images()

@client.command()
async def show(ctx):
	code: str = channel_to_code.get(ctx.channel.id)
	game: Game = current_games.get(code)
	if game is None:
		await ctx.send('Start a new game with "$newgame" first')
		return
	
	if game.player_prompts.keys() != game.player_images_b64.keys():
		await ctx.send('Still generating images...')
		game.show_on_finish = True
		return
	
	await game.show_next_image()

## run bot

TOKEN: str = os.getenv('DISCORD_TOKEN')
client.run(TOKEN)