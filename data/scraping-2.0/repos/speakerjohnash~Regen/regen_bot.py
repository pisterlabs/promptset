import os
import re
import random
import discord
from discord.ext import commands
from discord.ui import View, Button, TextInput, Modal
import pandas as pd
import openai

discord_key = os.getenv("CERES_DISCORD_BOT_KEY")
# local_discord_key = os.getenv("CERES_GOI_KEY")
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("REGEN_OPENAI_KEY")

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

models = {
	"ceres-old": "davinci:ft-regen-network-development-inc:ceres-stable-2022-12-11-22-17-03",
	"ceres-john": "davinci:ft-personal:ceres-refined-2022-10-21-02-46-56",
	"ceres-0": "ft:gpt-3.5-turbo-0613:regen-network-development-inc::7tPf5URV",
	"ceres-new": "ft:gpt-3.5-turbo-0613:regen-network-development-inc::7tknfSJC"
}

with open("ceres_system_instructions.txt", 'r') as file:
    system_instructions = file.read()

training_data = ""
people = []

class AskModal(Modal, title="Ask Modal"):

	answer = TextInput(label="Answer", max_length=400, style=discord.TextStyle.long)

	def add_view(self, question, view: View):
		self.answer.placeholder = question[0:100]
		self.view = view

	async def on_submit(self, interaction: discord.Interaction):
		embed = discord.Embed(title = "Your Response", description = f"\n{self.answer}")
		embed.set_author(name = interaction.user)
		await interaction.response.send_message(embed=embed)
		print(self.answer)
		self.view.stop()

def split_text_into_chunks(text, max_chunk_size=2000):
    """
    This function splits the input text into smaller chunks, each with a maximum size of max_chunk_size characters.
    It returns a list of text chunks, ensuring that the text is evenly distributed across the chunks and doesn't break mid-sentence.
    """

    # Calculate the number of chunks needed to evenly distribute the text
    num_chunks = max(1, (len(text) + max_chunk_size - 1) // max_chunk_size)

    # Adjust the chunk size to evenly distribute the text across the chunks
    chunk_size = (len(text) + num_chunks - 1) // num_chunks

    # Initialize variables
    text_chunks = []
    start_index = 0

    while start_index < len(text):
        end_index = start_index + chunk_size

        # Find the nearest sentence boundary before the end_index
        if end_index < len(text):
            boundary_index = text.rfind(".", start_index, end_index) + 1
            if boundary_index > start_index:  # If a boundary is found, update the end_index
                end_index = boundary_index

        # Add the chunk to the list of chunks
        text_chunks.append(text[start_index:end_index])

        # Update the start_index for the next iteration
        start_index = end_index

    return text_chunks

async def get_conversation_history(channel_id, limit, message_count, summary_count_limit):
	"""
	Fetches the conversation history from a specified Discord channel.
	The function retrieves a list of messages from the channel, ignoring slash commands and messages starting with '/'.
	"""

	channel = bot.get_channel(channel_id)
	messages = []
	summary_count = 0

	async for hist in channel.history(limit=limit):
		if not hist.content.startswith('/'):
			# Include embeds in the message content
			embed_content = "\n".join([embed.description for embed in hist.embeds if embed.description]) if hist.embeds else ""

			if hist.author == bot.user:
				summary_count += 1
				if summary_count < summary_count_limit:
					messages.append((hist.author, hist.content + embed_content))
			else:
				messages.append((hist.author, hist.content + embed_content))
			if len(messages) == message_count:
				break

	return messages

async def ceres_pool(message):
	"""
	Assists users in a Discord channel as Ceres
	"""

	channel_id = 988876280751616050
	# channel_id = 1105541234879111339
	channel = bot.get_channel(channel_id)

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]

	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 13, 11)
	messages.reverse()

	# Raw Ceres Answer
	ceres_answer = await n_shot(message)

	conversation = [
		{"role": "system", "content": "You are an interface to Ceres, a regenerative AI made by Regen Network to help teach people about Regen Network and how to regenate the planet"},
		{"role": "system", "content": "You are mediating a public thread on the Regen Network discord server"}
	]

	for m in messages:
		if m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"{m[0].name}: {m[1]}"})

	conversation.append({"role": "system", "content": "You asked Ceres via API the last prompt and Ceres said: " + ceres_answer})
	conversation.append({"role": "system", "content": "Answer the user. Prefence the speakers words over Ceres"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.5,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

def response_view(modal_text="default text", modal_label="Response", button_label="Answer"):	

	async def view_timeout():
		modal.stop()	

	view = View()
	view.on_timeout = view_timeout
	view.timeout = None
	view.auto_defer = True

	modal = AskModal(title=modal_label)
	modal.auto_defer = True
	modal.timeout = None

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	button = Button(label=button_label, style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)
	modal.add_view(modal_text, view)

	return view, modal

async def frankenceres(message, answer="", heat=0.11):

	"""
	Queries Frankenceres
	"""

	gregory = 644279763065634851
	john = 572900074779049984

	testers = [gregory, john]

	# Check if user is Gregory or John
	use_gpt4 = message.author.id in testers

	# Get Ceres One Shot Answer First
	ceres_answer = await n_shot(message)
	print(ceres_answer)

	if len(answer) > 0:
		ceres_answer = ceres_answer + " \n\n " + answer

	# Load Chat Context
	messages = []

	async for hist in message.channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.embeds:
				messages.append((hist.author, hist.embeds[0].description))
			else:
				messages.append((hist.author.name, hist.content))
			if len(messages) == 12:
				break

	messages.reverse()

	# Construct Chat Thread for API
	conversation = [{"role": "system", "content": "You are are a regenerative bot named Ceres that answers questions about Regen Network"}]
	conversation.append({"role": "user", "content": "Whatever you say be creative in your response. Never simply summarize, always say it a unique way but be brief and clear. I asked Ceres and she said: " + ceres_answer})
	conversation.append({"role": "assistant", "content": "I am speaking as Ceres. I was trained by Gregory Landua and Speaker John Ash. I will answer using Ceres as a guide as well as the rest of the conversation. Ceres said " + ceres_answer + " and I will take that into account in my response as best I can"})
	text_prompt = message.content

	for m in messages:
		if m[0] == bot.user:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": ceres_answer})
	conversation.append({"role": "user", "content": text_prompt})

	model_to_use = "gpt-4" if use_gpt4 else "gpt-3.5-turbo"

	response = openai.ChatCompletion.create(
		model=model_to_use,
		temperature=heat,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

async def n_shot(message, model="ceres-new", shots=5, heat=0):

	model_name = models[model]

	# Load Chat Context
	messages = []

	async for hist in message.channel.history(limit=50):
		if not hist.content.startswith('/') and hist.content.strip():
			if hist.embeds and hist.embeds[0].description is not None:
				messages.append((hist.author, hist.embeds[0].description))
			else:
				messages.append((hist.author.name, hist.content))
			if len(messages) == shots:
				break

	messages.reverse()

	# Construct Chat Thread for API
	conversation = [{"role": "system", "content": system_instructions}]

	for m in messages:
		if m[0] == bot.user:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	try:
		response = openai.ChatCompletion.create(
			model=model_name,
			messages=conversation,
			temperature=heat,
			max_tokens=256,
			top_p=1,
			frequency_penalty=1.1,
			presence_penalty=1.1
		)
		iris_answer = response.choices[0].message.content.strip()
	except Exception as e:
		print(f"Error: {e}")
		iris_answer = ""

	return iris_answer

def elaborate(ctx, prompt="prompt"):

	e_prompt = prompt + ". \n\n More thoughts in detail below. \n\n"

	button = Button(label="elaborate", style=discord.ButtonStyle.blurple)

	async def button_callback(interaction):

		if button.disabled:
			return

		button.disabled = True
		await interaction.response.defer()

		response = openai.Completion.create(
			model=models["ceres"],
			prompt=e_prompt,
			temperature=0.22,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()

		if len(response_text) == 0:

			response = openai.Completion.create(
				model=models["ceres"],
				prompt=e_prompt,
				temperature=0.8,
				max_tokens=222,
				top_p=1,
				frequency_penalty=1.7,
				presence_penalty=1.7,
				stop=["END"]
			)

			response_text = response.choices[0].text.strip()

		response_text = response_text.replace("###", "").strip()

		if len(response_text) == 0: response_text = "Ceres has no more to communicate after two requests"

		embed = discord.Embed(title = "Elaboration (beta)", description = f"**Prompt**\n{prompt}\n\n**Elaboration**\n{response_text}")

		await ctx.send(embed=embed)


	button.callback = button_callback

	return button

def load_training_data():

	global training_data

	try:
		training_data = pd.read_csv('ceres_training-data.csv')
	except:
		with open('ceres_training-data.csv', 'w', encoding='utf-8') as f:
			training_data = pd.DataFrame(columns=['prompt', 'completion', 'speaker'])
			training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

@bot.event
async def on_ready():
	load_training_data()
	print("Ceres is online")

@bot.event
async def on_close():
	print("Ceres is offline")

@bot.event
async def on_message(message):

	# Manage Ceres Pool
	if message.channel.id == 988876280751616050 and message.author != bot.user:
		await ceres_pool(message)
		await bot.process_commands(message)
		return

	# Handle DM Chat
	if not message.content.startswith("/") and isinstance(message.channel, discord.DMChannel) and message.author != bot.user:
		await frankenceres(message)

	await bot.process_commands(message)

@bot.command(aliases=['c'])
async def channel(ctx, *, topic=""):

	df = pd.read_csv('ceres_training-data.csv')
	prompts = df['prompt'].tolist()
	question_pattern = r'^(.*)\?\s*$'
	non_questions = list(filter(lambda x: isinstance(x, str) and re.match(question_pattern, x, re.IGNORECASE), prompts))

	random_non_question = random.choice(non_questions)
	message = ctx.message
	message.content = "Share a snippet of abstract and analytical wisdom related to the following topic. Be pithy: " + random_non_question

	await frankenceres(message, answer="")

@bot.command()
async def faq(ctx, *, topic=""):

	df = pd.read_csv('ceres_training-data.csv')
	prompts = df['prompt'].tolist()
	question_pattern = r'^(.*)\?\s*$'
	questions = list(filter(lambda x: isinstance(x, str) and re.match(question_pattern, x, re.IGNORECASE), prompts))
	questions = list(set(questions))

	question_completion_pairs = []

	# Iterate through each question and find its corresponding completions
	for question in questions:
		completions = df.loc[df['prompt'] == question, 'completion'].tolist()
		for completion in completions:
			question_completion_pairs.append((question, completion))

	# Remove any duplicate question-completion pairs from the list
	question_completion_pairs = list(set(question_completion_pairs))

	message = ctx.message
	random_question = random.choice(question_completion_pairs)
	embed = discord.Embed(title = "FAQ", description=random_question[0])
	message.content = random_question[0]

	await ctx.send(embed=embed)
	await frankenceres(message, answer=random_question[1])

@bot.command(aliases=['ask'])
async def ceres(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "Gregory | RND", "JohnAsh", "Dan | Regen Network"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	thought_prompt = thought + "\n\n###\n\n"

	response = openai.Completion.create(
		model=models["ceres"],
		prompt=thought_prompt,
		temperature=0.69,
		max_tokens=222,
		top_p=1,
		frequency_penalty=1.8,
		presence_penalty=1.5,
		stop=["END"]
	)

	text = response['choices'][0]['text']
	text = text.replace("###", "").strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")

	await ctx.send(embed=embed)

	# Send Clarification and Share UI
	view, modal = response_view(modal_text="Write your clarification here", modal_label="Clarification", button_label="feedback")
	el_prompt = thought + "\n\n" + text
	elaborate_button = elaborate(ctx, prompt=el_prompt)
	view.add_item(elaborate_button)
	await ctx.send(view=view)

	# Save Clarification
	await modal.wait()

	prompt = thought + "\n\n" + text

	if modal.answer.value is not None:
		training_data.loc[len(training_data.index)] = [prompt, modal.answer.value, ctx.message.author.name] 
		training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

@bot.command()
async def clarify(ctx, *, thought):
	"""
	/clarify send thourght to Gregory for clarification
	"""

	global training_data
	testers = ["John Ash's Username for Discord"]

	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	eve = 1005212665259495544
	gregory = 644279763065634851
	dan = 474842514407292930
	sja = 572900074779049984

	clarifiers = [dan, gregory]

	guild = bot.get_guild(989662771329269890)

	clarifier_accounts, modals = [], {}

	for clarifier in clarifiers:
		member = guild.get_member(clarifier)
		clarifier_accounts.append(member)
		question_embed = discord.Embed(title="Please clarify the below", description = thought)
		view, modal = response_view(modal_text=thought)
		modals[member.name] = modal
		sent_embed = discord.Embed(title = "Sent", description = f"Message sent for clarification")
		await member.send(embed=question_embed)
		await member.send(view=view)

	await ctx.send(embed=sent_embed)

	# Save Clarification
	for clarifier in modals:

		modal = modals[clarifier]

		await modal.wait()

		prompt = thought

		if modal.answer.value is not None:
			training_data.loc[len(training_data.index)] = [prompt, modal.answer.value, clarifier] 
			training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

@bot.command()
async def claim(ctx, *, thought):
	"""
	/claim log a claim for the iris to learn
	"""

	global training_data

	# Send Clarification and Share UI
	prompt = "Share something about in the general latent space of Regen Network"

	if thought is not None:
		training_data.loc[len(training_data.index)] = [prompt, thought, ctx.message.author.name] 
		training_data.to_csv('ceres_training-data.csv', encoding='utf-8', index=False)

	await ctx.send("Attestation saved")

@bot.command()
async def davinci(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "Gregory | RND", "JohnAsh", "Dan | Regen Network"]
	
	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	thought_prompt = thought

	response = openai.Completion.create(
		model="text-davinci-002",
		prompt=thought_prompt,
		temperature=0.69,
		max_tokens=222,
		top_p=1,
		frequency_penalty=1.8,
		presence_penalty=1.5,
		stop=["END"]
	)

	view = View()
	text = response['choices'][0]['text'].strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")

	await ctx.send(embed=embed)
	await ctx.send(view=view)

bot.run(discord_key)