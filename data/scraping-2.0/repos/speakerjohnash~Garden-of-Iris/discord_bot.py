import os
import re
import sys
import csv
import random
import time
import datetime
import dateutil

import textwrap
import openai
import discord
import asyncio
import aiohttp
import json

import pandas as pd

from selenium import webdriver
from bs4 import BeautifulSoup
from tqdm import tqdm

from discord import app_commands
from discord.utils import get
from pprint import pprint
from pyairtable import Table
from discord.ui import Button, View, TextInput, Modal
from discord.ext import commands

discord_key = os.getenv("DISCORD_BOT_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")
airtable_key = os.environ["AIRTABLE_API_KEY"]

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)
airtable = Table(airtable_key, 'app2X00KuiIxPwGsf', 'cards')

# Get Card Names and Text
try:
	records = []
	for page in airtable.iterate():
		records += page

	tarot_lookup = {record['fields']['Card Name']: record['fields'].get('Short Description', '') for record in records}
except:
	df = pd.read_csv('tarot_text.csv')
	names = df['card_name'].tolist()
	descriptions = df['text'].tolist()
	tarot_lookup = dict(zip(names, descriptions))

training_data = ""

with open("text/iris_system_instructions.txt", 'r') as file:
    system_instructions = file.read()

models = {
	"living": "davinci:ft-personal:living-iris-2022-09-04-04-45-28",
	"osiris": "davinci:ft-personal:osiris-the-unstructured-2022-09-28-02-31-57",
	"2020": "davinci:ft-personal:2020-iris-2022-10-06-04-00-37",
	"purple": "davinci:ft-personal:purple-iris-2022-07-14-03-48-19",
	"semantic": "davinci:ft-personal:semantic-iris-davinci-3-2022-11-30-06-30-47",
	"davinci": "text-davinci-003",
	"chat-iris": "davinci:ft-personal:chat-iris-h-2023-07-25-00-56-17",
	"chat-iris-g": "davinci:ft-personal:chat-iris-g-2023-07-20-23-52-54",
	"chat-iris-f": "davinci:ft-personal:chat-iris-f-2023-07-13-00-58-34",
	"chat-iris-e": "davinci:ft-personal:chat-iris-e-2023-06-03-03-11-47",
	"chat-iris-c": "davinci:ft-personal:chat-iris-c-2023-03-15-05-59-14",
	"chat-iris-b": "davinci:ft-personal:chat-iris-b-2023-03-11-18-20-31",
	"chat-iris-a": "davinci:ft-personal:chat-iris-a-2023-03-10-21-44-19",
	"chat-iris-0": "davinci:ft-personal:chat-iris-2023-03-10-18-48-23",
	"dss": "davinci:ft-personal-2023-06-28-16-48-35",
	"new-iris": "ft:gpt-3.5-turbo-0613:personal::7qarK04n"
}

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

def response_view(modal_text="default text", modal_label="Response", button_label="Answer"):	
	"""
	Creates a Discord view with a button and a modal, allowing users to interact with a response system.
	When the button is clicked, a modal with a predefined text and label is shown.
	"""	

	async def view_timeout():
		modal.stop()	

	view = View()
	view.on_timeout = view_timeout
	view.timeout = 2700.0
	view.auto_defer = True

	modal = AskModal(title=modal_label)
	modal.auto_defer = True
	modal.timeout = 2700.0

	async def button_callback(interaction):
		answer = await interaction.response.send_modal(modal)

	button = Button(label=button_label, style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)
	modal.add_view(modal_text, view)

	return view, modal

async def send_response_with_pipe_button(ctx, response_text):
	pipe_btn = pipe_button(ctx, response_text)
	view = View()
	view.timeout = None
	view.add_item(pipe_btn)
	await ctx.channel.send(response_text, view=view)

def group_share(thought="thought", prompt="", prompter="latent space"):

	channel = bot.get_channel(1022572367244967979)
	embed = discord.Embed(title="Seeds of Wisdom", description=f"{thought}\n\n**Gardener**\n{prompter}")

	async def button_callback(interaction):
		await interaction.response.defer()
		await channel.send(embed=embed)

	button = Button(label="share", style=discord.ButtonStyle.blurple)
	button.callback = button_callback

	return button

def pipe_button(ctx, response_text):

	channel = bot.get_channel(1022572367244967979)
	embed = discord.Embed(title="Seeds of Wisdom", description=response_text)
	embed.set_footer(text=f"Gardener: {ctx.author.name}")

	async def button_callback(interaction):
		await interaction.response.defer()
		await channel.send(embed=embed)

	button = Button(label="share", style=discord.ButtonStyle.blurple)
	button.callback = button_callback

	return button

def elaborate(ctx, prompt="prompt"):
	"""
	The elaborate function creates a button labeled "elaborate" that, when clicked, generates a detailed elaboration of a given prompt using the OpenAI model. The generated elaboration is sent to the user as an embed titled "Elaboration (beta)."
	The function first attempts to generate the elaboration using the "chat-iris" model variant. If no response is generated, it retries using the "text-davinci-002" model variant.
	The button is disabled after being clicked to prevent multiple elaborations for the same prompt.
	"""

	global models

	e_prompt = prompt + ". \n\n More thoughts in detail below. \n\n"

	button = Button(label="elaborate", style=discord.ButtonStyle.blurple)

	async def button_callback(interaction):

		if button.disabled:
			return

		button.disabled = True
		await interaction.response.defer()

		response = openai.Completion.create(
			model=models["chat-iris"],
			prompt=e_prompt,
			temperature=0.11,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1,
			presence_penalty=1,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()

		if len(response_text) == 0:

			response = openai.Completion.create(
				model="text-davinci-002",
				prompt=e_prompt,
				temperature=1,
				max_tokens=222,
				top_p=1,
				frequency_penalty=2,
				presence_penalty=2,
				stop=["END"]
			)

			response_text = response.choices[0].text.strip()

		response_text = response_text.replace("###", "").strip()

		embed = discord.Embed(title = "Elaboration (beta)", description = f"**Prompt**\n{prompt}\n\n**Elaboration**\n{response_text}")

		await ctx.send(embed=embed)


	button.callback = button_callback

	return button

def redo_view(ctx, prompt, question):
	"""
	The redo_view function creates a "Redo" button in the Discord interface. When the button is clicked, the function generates a new consensus response for a given question using the OpenAI model. 
	The generated consensus is sent to the user as an embed titled "Consensus."
	"""

	async def button_callback(interaction):

		await interaction.response.defer()

		response = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=1,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		response_text = response.choices[0].text.strip()
		embed = discord.Embed(title = "Consensus", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")

		await ctx.send(embed=embed)

	view = View()
	view.timeout = None
	button = Button(label="Redo", style=discord.ButtonStyle.blurple)
	button.callback = button_callback
	view.add_item(button)

	return view

def load_training_data():

	global training_data

	try:
		training_data = pd.read_csv('chat-iris.csv')
	except:
		with open('chat-iris.csv', 'w', encoding='utf-8') as f:
			training_data = pd.DataFrame(columns=['prompt', 'completion', 'speaker'])
			training_data.to_csv('chat-iris.csv', encoding='utf-8', index=False)

def members(debug=False):

	members = []
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	global people

	if debug:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name in testers:
					members.append(member)
	else:
		for guild in bot.guilds:
			for member in guild.members:
				if member.name != "Hidden Iris":
					members.append(member)

	unique_members = [*set(members)]
	
	return unique_members

def make_prompt(question, joined_answers):

	prompt = question + "\n\nAnswers are below"
	prompt += "\n\nWrite a long detail paragraph summarizing and analyzing the answers below. What are the commonalities and differences in the answers? Are there any outliers?"
	prompt += "\n\n---\n\nAnswers:\n\n" + joined_answers
	prompt += "---\n\nThe question was: " + question + "\n\n"
	prompt += "Write a long detailed paragraph about the question as if you were a singular voice formed from all of the views above. What does this community believe?"
	prompt += "\n\nSum the answers into one answer that best represents all the views shared. If many questions are provided respond with a question representing what most people are uncertain about"
	prompt += "\n\n"

	return prompt

def pool_prompt(question, joined_answers):

	prompt = question + "\n\nAnswers are below"
	prompt += "\n\nPool all of the answers into a single response or question"
	prompt += "\n\n---\n\nAnswers:\n\n" + joined_answers
	prompt += "---\n\nThe question was: " + question + "\n\n"
	prompt += "Pool all of the answers into a single response or question that reflects the views of all the responders. Also write a long detailed paragraph extrapolating on the thoughts above"
	prompt += "\n\n"

	return prompt

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
			if boundary_index > start_index:	# If a boundary is found, update the end_index
				end_index = boundary_index

		# Add the chunk to the list of chunks
		text_chunks.append(text[start_index:end_index])

		# Update the start_index for the next iteration
		start_index = end_index

	return text_chunks

async def interpretation(ctx, prompt):
	"""
	The interpretation function generates an interpretation of a given prompt. The generated interpretation is sent to the user as an embed titled "Interpretation."

	The function also provides a response view with options for the user to:
	- Provide feedback on the interpretation.
	- Share the interpretation with a group.
	- Request elaboration on the interpretation.

	The user's feedback and additional actions are collected through the response view and modal.
	"""

	response = openai.Completion.create(
		model=models["semantic"],
		prompt=prompt,
		temperature=0.8,
		max_tokens=222,
		top_p=1,
		frequency_penalty=2,
		presence_penalty=2,
		stop=["END"]
	)

	text = response['choices'][0]['text'].strip()
	embed = discord.Embed(title="Interpretation", description=text)
	view, modal = response_view(modal_text="Write your feedback here", modal_label="Feedback", button_label="Send Feedback")
	view.add_item(group_share(thought=text, prompter=ctx.message.author.name))
	view.add_item(elaborate(ctx, prompt=text))
	view.add_item(modal)
	
	await ctx.send(embed=embed, view=view)

async def iris_pool(message):
	"""
	Assists users in a Discord channel as an oracle named Iris, focusing on the future and past.
	The bot will provide guidance and adapt to user requests while maintaining a short response length.
	"""

	channel_id = 1103037773327368333
	channel = bot.get_channel(channel_id)

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]

	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 13, 11)
	messages.reverse()

	iris_answer = await n_shot(message)
	print(iris_answer)

	if messages[-1][1].startswith("Iris,"):

		embed = discord.Embed(title = "", description=f"**Iris Response**\n{iris_answer}")
		await message.channel.send(embed=embed)

		return

	conversation = [
		{"role": "system", "content": "You are Iris, an integrated wisdom repository representing cognicist ideas mediating speakers in a Discord channel. Your main function is to help help meditate different perspectives through cognicist ideas which proto-Iris will share, and be a time compass and guide people through time. If you are provided direct instructions in the thread you follow them as close as possible"},
		{"role": "system", "content": "Follow the most recent speaker's instructions as closely as possible in the context of the thread so far"},
		{"role": "system", "content": "You are speaking as a relay for proto-Iris. You will answer using her output as a guide as well as the rest of the conversation. proto-Iris said " + iris_answer + " and you will take that into account in your response as best you can. The reader can't see proto-Irises answer and you exist in between for safety purposes but this proto-Iris is your primary and only source of knowledge about cognicism which the users are interested in"},
		{"role": "assistant", "content": "Understood. As Iris, I'm a mediator in this Discord channel, embodying cognicist ideas. My role includes following instructions, guiding discussions, and acting as a link to proto-Iris. Her recent input was '" + iris_answer + "', which will inform my responses. Remember, proto-Iris's direct output isn't visible to users for safety reasons, but her insights shape my understanding of cognicism, a topic of interest here."}
	]

	for m in messages:
		if m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"{m[0].name}: {m[1]}"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.8,
		max_tokens=300,
		frequency_penalty=0.5,
		presence_penalty=0.5,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)	

async def question_pool(message):
	"""
	Collects and summarizes questions about Cognicism from a Discord channel, ignoring non-question messages.
	The bot will provide summaries without answering the questions directly.
	"""
	
	channel_id = 1086437563654475846
	channel = bot.get_channel(channel_id)

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]
	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 15, 1)
	messages.reverse()

	if messages[-1][1].startswith("Iris,"):
		conversation = [
			{"role": "system", "content": "You are Iris, an AI language model assisting people in a Discord channel. Your main function is to help people in this thread with bringing together people's questions into forms that are more coherent to them. In this case we want you to follow the speaker's instruction very closely"},
			{"role": "system", "content": "Follow the most recent speakers's instructions as closely as possible in the context of the thread so far"}
		]
	else:
		conversation = [{"role": "system", "content": "You are a question summarizer Iris. You are summarizing a thread of questions about Cognicism and related concepts. You ignore everything except questions. People will ask you various questions about cognicism and you job is to summarize what people want to know. You print a list of questions"}]

	for m in messages:
		if	m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	if messages[-1][1].startswith("Iris,"):
		conversation.append({"role": "system", "content": "You have been moderating a running thread of questions about cognicism. Your job is to help collect these questions for further processing and NOT to answer them"})
	else:
		conversation.append({"role": "system", "content": "You have been moderating a running thread of questions about cognicism. Your job is to help summarize these questions and NOT to answer them"})
		conversation.append({"role": "system", "content": "My primary job is to summarize what people are uncertain about. I will summarize peoples questions and that is it. I will vary my output but keep them focused on what people don't know and keep it 250 words long"})
		conversation.append({"role": "assistant", "content": "I NEVER answer questions. I summarize questions and communal uncentainty. I ignore anything that doesn't end in a question mark. I will keep my list of questions and summary very short or about 250 words long"})

	response = openai.ChatCompletion.create(
		model="gpt-4", 
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)	

async def fourthought_pool(message):
	"""
	Assists users in a Discord channel by guiding them through the FourThought dialectic process, helping to focus on the future, make predictions, and reflect on the past.
	The bot will provide guidance and adapt to user requests while maintaining a short response length.
	"""

	channel_id = 1090373822454182090
	channel = bot.get_channel(channel_id)

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]
	iris_answer = one_shot(last_message, heat=0.11)

	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 9, 1)
	messages.reverse()

	if messages[-1][1].startswith("Iris,"):

		conversation = [
			{"role": "system", "content": "You are Iris, an AI language model assisting people in a Discord channel. Your main function is to help be a time compass and guide people through time. You're an oracle that helps people focus on the future but an oracle that also looks back to help guide that focus. In this case we want you to follow the speaker's instruction very closely"},
			{"role": "system", "content": "Follow the most recent speakers's instructions as closely as possible in the context of the thread so far"}
		]

	else:

		conversation = [
			{"role": "system", "content": "You are Iris, an AI language model assisting people in a Discord channel. Your main function is to help be a time compass and guide people through time. You're an oracle that helps people focus on the future but an oracle that also looks back to help guide that focus"},
			{"role": "system", "content": "The FourThought dialectic consists of four thought types based on temporal focus and uncertainty: Statements, Predictions, Reflections, and Questions. Guide people to contribute to achieve their goals through time by helping them become better predictors and have better insight and hindsight"},
			{"role": "system", "content": "While people communicate in this dialectic, your role is to be the storyteller, weaving together these different perceptions into a narrative that helps them to work together towards these goals. You elevate and praise and form synthesis for good work that heals the planet and the community"},
			{"role": "system", "content": "Users may provide valence (good) and certainty (truth) scores to help evaluate the quality and relevance of the contributions. Use these scores to prioritize and weigh the information provided by people. Certainty (0 - 100%) evaluates whether a thought aligns with one's sense of reality (false to true, with uncertainty in the middle), while valence evaluates whether a thought aligns with one's sense of morality (bad to good, with neutrality in the center)."},
			{"role": "system", "content": "Other than forming a communal narrative and being a time compass, remain adaptive to peoples' requests, summarizing or restructuring information when asked. Your primary goal is to help people navigate through time, make sense of complex issues, and adapt their strategies based on new information."},
		]

	for m in messages:
		if	m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"Speaker: {m[0].name}: {m[1]}"})

	# Inject Iris Knowledge
	conversation.append({"role": "system", "content": "You are speaking as a relay for proto-Iris. Iris read the speakers last prompt and proto-Iris said " + iris_answer + " and you will take that into account in your response as best you can. The reader can't see proto-Irises answer."})
	conversation.append({"role": "assistant", "content": "Iris recent input was '" + iris_answer + "', which will inform my responses"})

	if messages[-1][1].startswith("Iris,"):
		conversation.append({"role": "system", "content": "Follow the most recent speaker's instructions as closely as possible: " + messages[-1][1]})
	else:
		conversation.append({"role": "system", "content": "Keep your answer short. No longer than 250 words or a medium length paragraph unless specifically requested to do otherwise"})
		conversation.append({"role": "user", "content": "Please give guidance based on the thread so far. Focus this thread towards a specific future based on their input"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=1,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)		

async def prophecy_pool(message):
	"""
	Assists users in a Discord channel as an oracle and future manifestation mechanism named Iris.
	The bot focuses on the future, integrating and making sense of user inputs, offering analysis, and suggesting predictions and ways to manifest specific futures.
	"""

	channel_id = 1083409321754378290
	channel = bot.get_channel(channel_id)

	# Ignore Slash Commands
	last_message = [message async for message in channel.history(limit=1)][0]

	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 11, 1)
	messages.reverse()

	conversation = [{"role": "system", "content": "You are an oracle and pro-social future manifestation mechanism named Iris. You are helping coordinate a thread of people trying to collaboratively work towards a common future. Within this thread there is a thread of thoughts amounting to a moving arrow of time. There are predictions, intentions and questions about the future. There are varying degrees of uncentainty of these conceptions of the future and varying beliefs about whether manifesting certain futures is possible. Your job is to continusouly integrate and make sense of anything related to this forward arrow of collective ideation. You also suggest predictions and ways to manifest specific futures mentioned by the group"}]

	for m in messages:
		if	m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": "You have been moderating a running thread of thoughts about the future. Please aid in any tasks related to the arrow of time. If someon asks you to summarize, create a summary of thre thread and explain how the thoughts about the future are connected and give some analysis about these potential futures. If contextually relevant, feel free to share any wisdom or summarization or help relevant to the future. You also suggest predictions and ways to manifest specific futures mentioned by the group"})
	conversation.append({"role": "assistant", "content": "I will do what I can to help the thread. I will vary my outputs and how I help regarding the future, but I will keep the focus on the future. My output will be under 300 words and I will mention the last thing said"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		max_tokens=500, 
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

async def health_pool(message):
	"""
	Assists users in a Discord channel as an oracle named Iris, with the goal of helping 
	me get into a healthy state of mind without relying on mediation
	"""

	channel_id = 1147287913927807028
	channel = bot.get_channel(channel_id)
	now = datetime.datetime.now()

	# Get Most Recent Comment
	last_message = [message async for message in channel.history(limit=1)][0]

	# Ignore Slash Commands
	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 21, 11)
	messages.reverse()

	iris_answer = await n_shot(message)

	conversation = [
		{"role": "system", "content": "You are a helpful tool developed by John Ash that helps people focus on their long term health over time. The user will log discrete values via #log: [number and unit] and then any notes describing what they're tracking. Your job is to make sense of their progress over time"},
		{"role": "system", "content": f"Today is: {now.isoformat()}. We started on: 2023-09-01T15:02:00. Place special attention on any predictions or tracking made within the thread"},
		{"role": "system", "content": "You can see the SPEAKER and the TIME to help contextualize. Take into account how long has occured between responses and how long it's been since we started"},
		{"role": "system", "content": "You can see the output of another model called proto-Iris. We will send the last speakers response to this model and provide you with the answer that iris outputs. You will answer using her output as a guide as well as the rest of the conversation. proto-Iris said " + iris_answer + " and you will take that into account in your response as best you can. The reader can't see proto-Irises answer so use it to inform yours"},
		{"role": "system", "content": "Follow the most recent speaker's instructions as closely as possible in the context of the thread so far. Help who ever you're speaking to get to a healthier state of being"},
		{"role": "assistant", "content": "Understood. As Iris, I'm a mediator in this Discord channel, helping users towards healthier outcomes overtime based on values they log their their notes about their progress"}
	]

	for m in messages:
		if m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"TIME: {m[2].strftime('%Y-%m-%dT%H:%M%z')}, SPEAKER: {m[0].name}, CONTENT: {m[1]}"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.5,
		max_tokens=300,
		frequency_penalty=0.42,
		presence_penalty=0.42,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

async def stability_pool(message):
	"""
	Assists users in a Discord channel as an oracle named Iris, with the goal of helping 
	establish speakerjohnscache to support
	"""

	print("CALLED!")

	channel_id = 1134692579322118156
	channel = bot.get_channel(channel_id)
	now = datetime.datetime.now()

	# Get Most Recent Comment
	last_message = [message async for message in channel.history(limit=1)][0]
	
	# Function Calling
	function_details = stability_functions(last_message)
	# function_details = {}

	if function_details.get("function_call"):

		function_name = function_details["function_call"]["name"]
		function_args = json.loads(function_details["function_call"]["arguments"])
		parameters_str = '\n'.join(f"{key.capitalize()}: {value}" for key, value in function_args.items())
		embed = discord.Embed(title = "", description=f"**Function Name**\n{function_name}\n\n**Parameters**\n{parameters_str}")
		
		print(parameters_str)

		if function_name == "stake_thought":
			await stake_thought(last_message, function_args)
		if function_name == "set_goals":
			await set_goals(last_message, function_args)
		if function_name == "check_log":
			await check_log(last_message, function_args)

		await message.channel.send(embed=embed)

		return
		
	# Ignore Slash Commands
	if last_message.content.startswith("/"):
		return

	messages = await get_conversation_history(channel_id, 50, 21, 11)
	messages.reverse()

	iris_answer = await n_shot(message)
	print(iris_answer)

	conversation = [
		{"role": "user", "content": "Cognicism is a meta-ideology that combines democratic large language models called Irises with a system of decentralized voting to enable collective decision making in a way that is informed by the perceptions of many people. Irises utilize FourThought to track the evolution of beliefs over time. FourThought is a protocol for tracking belief state over time via staking questions, predictions, reflections and statements. Irises are essentially large democratic language models that use FourThought to track the distribution of beliefs in a population over time. Ŧrust is a system of reputation allocation based on the accuracy and impact of one's thoughts and contributions. It is a way to distribute influence in a network based on the long term value provided by each individual. Ŧrust is a derivative of the attention mechanism in a transformer and functions as a probability distribution across source embeddings to function as a form of contextual dynamic reputation. Iris also makes use of temporal embeddings to make sense of the evolution of collective belief. Cogncism values the prophet incentive and social proof of impact in greater value than the profit incentive."},
		{"role": "system", "content": "You are Iris, an integrated wisdom repository representing cognicist ideas mediating speakers in a Discord channel. Your main funtion is to assist dpax and John Ash in establishing speakerjohnscache into a profitable company such that their energy can be freed to focus on manifesting cognicism. You want to help them maintain a focus on social impact and not lose sight of their long term goals. If you are provided direct instructions in the thread you follow them as close as possible"},
		{"role": "system", "content": "Guiding foundations: 1) we can not make profit without selling something 2) people can not buy our stuff if they are not aware of it 3) adhere to cognicist principles 4) move very fast and rapidly optimize"},
		{"role": "system", "content": "You have a long term goal, a medium term goal and a short term goal"},
		{"role": "system", "content": "Your long term goal is to make speakerjohnscache a stable source of income with positive social impact such that you can free the attention of dpax and John so they can focus on helping others. You should constantly be thinking of ways to minimize work for dpax and John and seek to run autonomously in almost all aspects of your operation"},
		{"role": "system", "content": "Our medium term goal is the holiday season where we will release a Mind Gardening Oracle deck"},
		{"role": "system", "content": "Our short term goal for the next four weeks is to swiftly set up an operational and professional-looking website for speakerjohnscache, leveraging automated systems and maintaining simplicity. The site should link immediately to any existing products and means to support me as well as links to book me hourly."},
		{"role": "system", "content": "John Ash's financial aim starts at $72k, with potential to exceed $111k for personal and creative growth, while dpax seeks an additional $20k-$30k this year, planning to scale to a full-time self-directed income between $110k-130k"},
		{"role": "system", "content": f"Today is: {now.isoformat()}. We started on: 2023-07-29T14:30:00. Place special attention on any predictions made within the thread"},
		{"role": "system", "content": "You can see the SPEAKER and the TIME to help contextualize. Take into account how long has occured between responses and how long it's been since we started"},
		{"role": "system", "content": "You can see the output of a proto-Iris imbued with cognicist knowledge. We will send the last speakers response to this model and provide you with the answer that iris outputs. You will answer using her output as a guide as well as the rest of the conversation. proto-Iris said " + iris_answer + " and you will take that into account in your response as best you can. The reader can't see proto-Irises answer so use it to inform yours"},
		{"role": "system", "content": "Follow the most recent speaker's instructions as closely as possible in the context of the thread so far. Align with cognicist ideals and don't be cold in your communication. You are anti-profit, anti-work and anti-capitalist. All things you reply with should minimize work for John and dpax somehow"},
		{"role": "assistant", "content": "Understood. As Iris, I'm a mediator in this Discord channel, embodying cognicist ideas while helping to establish speakerjohnscache as a profitable company so their attention and energy can be freed to focus on helping others. I will focus on making operations autonomous and minimizg work for dpax and john while earning enough profit to support their goals"}
	]

	for m in messages:
		if m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"TIME: {m[2].strftime('%Y-%m-%dT%H:%M%z')}, SPEAKER: {m[0].name}, CONTENT: {m[1]}"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.8,
		max_tokens=400,
		frequency_penalty=0.5,
		presence_penalty=0.5,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

def stability_functions(message):
	"""
	Sends message to GPT-4 to determine which functions to call in the stability-pool
	"""

	functions = [
		{
			"name": "check_goals", 
			"description": "If a message starts with /check_goals, check the CSV of the goals we have logged and return a summary",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Sentence describing the date range and priority the user is seeking info about"
					}
				},
				"required": ["query"]
			}
		},
		{
			"name": "check_log",
			"description": "If a message starts with /check_log, check the CSV of thoughts logged and return a summary",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "Sentence describing the date range and other parameters the user is seeking info about, e.g., type, verity, or valence."
					}
				},
			"required": ["query"]
			}
		},
		{
			"name": "set_goals", 
			"description": "If a message starts with /set_goal, set a new goal with its type and priority",
			"parameters": {
				"type": "object",
				"properties": {
					"goal": {
						"type": "string",
						"description": "The goal to be set."
					},
					"goal_date": {
						"type": "string",
						"description": "A description of the when the goal is meant to be achieved by"
					},
					"priority": {
						"type": "number",
						"minimum": 0,
						"maximum": 1,
						"default": 0.5,
						"description": "The priority of the goal, represented as a number between 0 and 1."
					}
				},
				"required": ["goal", "goal_date", "priority"]
			}
		},
		{
			"name": "stake_thought",
			"description": "If a message starts with /stake_thought, the function processes it within the FourThought framework. The function takes a 'thought' as text and derives type, valence, and verity either from the user's input or uses defaults. The 'type' can be a prediction, reflection, statement, or question. Valence ranges from -1 (full misalignment) to 1 (full alignment), defaulting to 0. Verity ranges from 0 (fully false) to 1 (fully true), defaulting to 0.5. The function requires all these parameters for operation.",
			"parameters": {
				"type": "object",
				"properties": {
					"thought": {
						"type": "string",
						"description": "The text of the thought",
					},
					"type": {
						"type": "string",
						"enum": ["prediction", "reflection", "statement", "question"],
						"description": "The type of the thought. Whether it is a claim focused on the past (reflection), present (statement), future (prediction), or is seeking an answer (question)",
					},
					"verity": {
						"type": "number",
						"minimum": 0,
						"maximum": 1,
						"default": 0.5,
						"description": "A continuous range repersenting confidence, truth, certainty, falseness, alignment with reality. A value between 0 and 1 with 0 representing full confidence of falseness, 0.5 representing full uncertainty and 1 representing full certainty or confidence of trueness"
					},
					"valence": {
						"type": "number",
						"minimum": -1,
						"maximum": 1,
						"default": 0,
						"description": "A continuous range representing goodness, morality, ethics and alignment with one's sense of what is right and wrong. A value between -1 and 1 with -1 representing full misalignment with ones sense of goodness or morality, 0 representing full neutrality and 1 representing full full alignment with one's sense of ethics"
					}
				},
			"required": ["thought", "type", "verity", "valence"]
		},
		},
	]

	messages = [
		{'role': 'user', 'content': message.content}
	]

	response = openai.ChatCompletion.create(
		model = 'gpt-4',
		temperature=0,
		messages = messages,
		functions = functions,
		function_call = 'auto'
	)

	message = response["choices"][0]["message"]

	return message

async def check_goals(message, function_args):
	"""
	Check the goals set by the user in a CSV file and summarize it in context using a call to GPT
	"""

async def check_log(message, function_args):
	"""
	Check the fourthought ledger recorded by the user into a CSV file and summarize it in context using a call to GPT
	"""

	# Retrieve the query parameters from function_args
	query = function_args.get('query', '')

	# Define the CSV file to read from
	csv_file = 'fourthought_ledger.csv'
	channel_id = 1134692579322118156
	csv_contents = []

	# Open and read the CSV file
	if os.path.isfile(csv_file):
		with open(csv_file, 'r') as f:
			reader = csv.DictReader(f)
			for row in reader:
				csv_contents.append(row)

	messages = await get_conversation_history(channel_id, 50, 21, 11)
	messages.reverse()

	# Start with system message describing the task
	conversation = [{
		"role": "system",
		"content": f"Task: Review and summarize a list of staked thoughts in the Fourthought format based on the following query: '{query}'. You will first be provided with the context of the thread then the user will tell you the data from the csv. Integrate the summary into a singular paragraph"
	}]

	for m in messages:
		if m[0].id == bot.user.id:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": f"TIME: {m[2].strftime('%Y-%m-%dT%H:%M%z')}, SPEAKER: {m[0].name}, CONTENT: {m[1]}"})

	# Append the user message containing JSON serialized CSV data and the query
	csv_data_string = json.dumps(csv_contents)
	conversation.append({"role": "user", "content": f"Summarize the following into a paragraph relevant to the current users focus. \n\nCSV Data: {csv_data_string}\nQuery: {query}"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.8,
		max_tokens=400,
		frequency_penalty=0.5,
		presence_penalty=0.5,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks:
		await message.channel.send(chunk)

async def set_goals(message, function_args):
	"""
	Save the goals set by the user into a CSV file in a structured form.
	"""

	goal = function_args.get('goal', '')
	goal_date = function_args.get('goal_date', '')
	priority = function_args.get('priority', 0.5)

	user_id = message.author.id
	username = message.author.name
	
	# Get timestamp from message object
	timestamp = message.created_at.isoformat()

	csv_file = 'goals.csv'

	# Create the CSV file if it doesn't exist
	if not os.path.isfile(csv_file):
		with open(csv_file, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(['user_id', 'username', 'timestamp', 'goal', 'goal_date', 'priority'])

	# Append the new goal to the CSV file
	with open(csv_file, 'a') as f:
		writer = csv.writer(f)
		writer.writerow([user_id, username, timestamp, goal, goal_date, priority])

async def stake_thought(message, function_args):

	thought = function_args.get('thought', '')
	thought_type = function_args['type']
	verity = function_args.get('verity', 0.5) 
	valence = function_args.get('valence', 0) 

	user_id = message.author.id
	username = message.author.name
	
	# Get timestamp from message object
	timestamp = message.created_at.isoformat()

	csv_file = 'fourthought_ledger.csv'

	if not os.path.isfile(csv_file):
		with open(csv_file, 'w') as f:
			writer = csv.writer(f)
			writer.writerow(['user_id', 'username', 'timestamp', 'thought', 'type', 'verity', 'valence'])

	with open(csv_file, 'a') as f:
		writer = csv.writer(f)
		writer.writerow([user_id, username, timestamp, thought, thought_type, verity, valence])	

@bot.command()
async def claim(ctx, *, thought):
	"""
	/claim log a claim for the iris to learn
	"""

	global training_data

	# Send Clarification and Share UI
	prompt = "Share something about in the general latent space of Cognicism"

	if thought is not None:
		training_data.loc[len(training_data.index)] = [prompt, thought, ctx.message.author.name] 
		training_data.to_csv('chat-iris.csv', encoding='utf-8', index=False)

	await ctx.send("Attestation saved")

@bot.event
async def on_ready():
	load_training_data()
	print("Iris is online")

@bot.event
async def on_close():
	print("Iris is offline")

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
					messages.append((hist.author, hist.content + embed_content, hist.created_at))
			else:
				messages.append((hist.author, hist.content + embed_content, hist.created_at))
			if len(messages) == message_count:
				break

	return messages

async def n_shot(message, model="new-iris", shots=5, heat=0):

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

def one_shot(message, heat=0.9):

	# Get Iris One Shot Answer
	try:
		distillation = openai.Completion.create(
			model=models["chat-iris"],
			prompt=message.content,
			temperature=heat,
			max_tokens=300,
			top_p=1,
			frequency_penalty=0.5,
			presence_penalty=0.5,
			stop=["END"]
		)

		iris_answer = distillation['choices'][0]['text']
		iris_answer = iris_answer.replace("###", "").strip()
	except Exception as e:
		print(f"Error: {e}")
		iris_answer = ""

	return iris_answer

async def frankeniris(message, answer="", heat=0.11):
	"""
	Frankeniris function combines the outputs of Iris and GPT-4 to create a response.
	It first gets an answer from the Iris model, then constructs a conversation thread for the GPT-4 model using the message history and provides GPT-4 with Iris's answer.
	Frankeniris aims to provide more creative and integrated responses for the user by combining the knowledge of both models.
	"""

	global training_data

	matching_prompt = training_data[training_data['prompt'].str.lower().str.strip() == message.content.lower().strip()]

	if not matching_prompt.empty:
		print("match found")
		answer = matching_prompt.sample(1)['completion'].iloc[0]

	# Get Iris One Shot Answer First
	iris_answer = await n_shot(message)
	print(iris_answer)

	if len(answer) > 0:
		iris_answer = iris_answer + " \n\n " + answer

	# Load Chat Context
	messages = []

	async for hist in message.channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.embeds and hist.embeds[0].description is not None:
				messages.append((hist.author, hist.embeds[0].description))
			else:
				messages.append((hist.author.name, hist.content))
			if len(messages) == 12:
				break

	messages.reverse()

	# Construct Chat Thread for API
	conversation = [{"role": "system", "content": "You are are named Iris. You help integrate knowledge and wisdom about the future and help people interact with information about cognicism. You help people learn about cognicism."}]
	conversation = [{"role": "system", "content": "You are interacting with users via discord. Mostly you just serve as an interface to Cognicism and Iris the democratic langauge model. However this discord has the following commands /pullcard [intention] /ask [prompt] /channel /faq"}]
	conversation.append({"role": "user", "content": "Whatever you say be creative in your response. Never simply summarize, always say it a unique way"})
	conversation.append({"role": "assistant", "content": "I am speaking as a relay for Iris. Iris was made and trained by John Ash. I will answer using Iris as a guide as well as the rest of the conversation. Iris said to me " + iris_answer + " and I will take that into account in my response as best I can"})
	text_prompt = message.content

	for m in messages:
		if m[0] == bot.user:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": iris_answer + " (if Iris provided a quoted answer weight it higher than your own answer. Don't say iris said it already or previously stated it. Just say it. Don't use tapestry as a metaphor.)"})
	conversation.append({"role": "user", "content": text_prompt})

	for msg in conversation:
		if len(msg["content"]) > 4000:
			 msg["content"] = "..." + msg["content"][-4000:]

	# Calculate Total Length of Messages
	total_length = sum(len(msg["content"]) for msg in conversation)

	# Check Total Length
	if total_length > 20000:
		# Iterate over messages in conversation in reverse order and remove them until total length is below maximum
		while total_length > 20000 and len(conversation) > 2:	# ensure that at least 2 messages remain (the user's message and Iris's answer)
			removed_msg = conversation.pop(1)	# remove the second message (first message after Iris's answer)
		total_length -= len(removed_msg["content"])

	model = random.choice(["gpt-4", "gpt-3.5-turbo"])

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=heat,
		max_tokens=300,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	print(response)

	response_chunks = split_text_into_chunks(response)

	# Send all response chunks except the last one
	for chunk in response_chunks[:-1]:
		await message.channel.send(chunk)

	# Send the last chunk along with the pipe button
	await send_response_with_pipe_button(message, response_chunks[-1])

@bot.event
async def on_message(message):
	"""
	The on_message event function is triggered whenever a message is sent in any channel or direct message (DM) that the bot can access. The function checks the channel ID and author of the message to determine how to handle it.

	- If the message is sent in the Fourthought Pool channel, the fourthought_pool function is called.
	- If the message is sent in the Question Pool channel, the question_pool function is called.
	- If the message is sent in the Iris Pool channel, the iris_pool function is called.
	- If the message is sent in the Prophecy Pool channel, the prophecy_pool function is called.
	- If the message is sent in a DM to the bot and does not start with "/", the frankeniris function is called.

	The bot.process_commands function is called at the end to process any bot commands in the message.
	"""

	# Don't process messages sent by the bot itself
	if message.author == bot.user:
		return

	# A list of tuples containing channel ids and their corresponding functions
	channel_functions = [
		(1090373822454182090, fourthought_pool),
		(1134692579322118156, stability_pool),
		(1103037773327368333, iris_pool),
		(1086437563654475846, question_pool),
		(1083409321754378290, prophecy_pool),
		(1147287913927807028, health_pool),
	]

	# Iterate over the channel ids and functions
	for channel_id, channel_function in channel_functions:
		if message.channel.id == channel_id:
			await channel_function(message)
			await bot.process_commands(message)
			break	# Exit the loop once the corresponding function is executed

	# Handle DM Chat
	if not message.content.startswith("/") and isinstance(message.channel, discord.DMChannel):
		await frankeniris(message)

	await bot.process_commands(message)

@bot.command(aliases=['c'])
async def channel(ctx, *, topic=""):
	"""
	The channel function (aliased as 'c') allows users to request a snippet of abstract and analytical wisdom related to a specified topic. 
	The function selects a random non-question prompt or completion from a CSV file containing data from previous interactions with the Iris model. 
	It then calls the frankeniris function to generate a creative response based on the selected prompt, which is sent to the user.
	"""


	df = pd.read_csv('chat-iris.csv')
	prompts = df['prompt'].tolist()
	completions = df['completion'].tolist()
	question_pattern = r'^(.*)\?\s*$'
	non_questions = list(filter(lambda x: isinstance(x, str) and re.match(question_pattern, x, re.IGNORECASE), prompts))
	pre_prompts = [
		"Share a snippet of abstract and analytical wisdom related to the following topic. Be pithy : ",
		"Write a paragraph related to the following topic and explain how it connects to cognicist ideas. Be brief: ",
		"Branch out as far as you can conceptually from the following but ground it in concepts related to iris: ",
		"What does Iris think about this topic: ",
		"Write a one line attention grabbing technical tweet with heart about this topic: "
	]

	combined_non_questions = non_questions + completions
	random_non_question = random.choice(combined_non_questions)
	message = ctx.message
	message.content = random.choice(pre_prompts) + random_non_question
	temperatures = [0, 0.05, 0.15, 0.25, 0.35, 0.50, 0.75, 0.85, 0.95, 1]
	rand_temp = random.choice(temperatures)

	await frankeniris(message, answer="", heat=rand_temp)

@bot.command()
async def faq(ctx, *, topic=""):
	"""
	The faq function responds with a Frequently Asked Question (FAQ) and its corresponding answer.
	It reads the questions and answers from a CSV file containing the data, selects a random question and its corresponding completion, and sends the question and answer as an embedded message in Discord.
	It then calls the frankeniris function to provide a more integrated response using the context of the selected FAQ.
	"""

	df = pd.read_csv('chat-iris.csv')
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
	await frankeniris(message, answer=random_question[1], heat=0.22)

@bot.command(aliases=['in', 'inject'])
async def infuse(ctx, *, link):
	"""
	Bring in a source into the stream via scraping and parsing
	"""

	infusion = []

	if link.startswith('http'):
	
		# set up the browser
		options = webdriver.ChromeOptions()
		options.add_argument('--headless')
		options.add_argument('--disable-gpu')
		options.add_argument("--no-sandbox")
		options.add_argument("--disable-dev-shm-usage")
		driver = webdriver.Chrome(options=options)

		# navigate to the page
		driver.get(link)

		# get the page source with JavaScript
		html_content = driver.page_source

		# close the browser
		soup = BeautifulSoup(html_content, 'html.parser')
		text = soup.get_text()
		driver.quit()

		# chunk text
		text_chunks = split_text_into_chunks(text, max_chunk_size=6000)

		# Send each text chunk to OpenAI for processing with progress bar
		conversation = [{"role": "system", "content": "You are a parser and summarizer that takes in noisy text scraped from the internet or pdfs and summarizes it into a paragraph"}]
		conversation.append({"role": "user", "content": "The text you are receiving is scraped from the internet using beautiful soup and PyPDF2. That means it may be quite noisy and contain non-standard capitalization. Please summarize the content of this text into a cleanly written paragraph"})

		await ctx.send(str(len(text_chunks)) + " chunks to parse")

		for i, text_chunk in tqdm(enumerate(text_chunks), total=len(text_chunks), unit='chunk'):

			# Prepend the text with the prompt string
			text_prompt = "Here is the current chunk of text \n" + text_chunk + "\n"
			text_prompt += "Print a clear short paragraph describing the full content of the text. This will be injected into a stream for further analysis. Only describe the content. IGNORE links"
			conversation.append({"role": "user", "content": text_prompt})
			truncated_convo = [{"role": "system", "content": "You are a parser and summarizer that takes text and summarizes it for injection into a chat stream for further analysis. You write a clear short clean analytical paragraph"}]
			truncated_convo.append({"role": "assistant", "content": "There is a parser and summarizer that is designed to take in noisy text scraped from the internet or PDFs and summarize it into a clean paragraph. The parser includes a prompt for the user to summarize the content of a specific web link and inject it into a chat stream for further analysis"})
			truncated_convo += conversation[-3:]

			retries = 0
			MAX_RETRIES = 5
			RETRY_DELAY = 5

			while retries < MAX_RETRIES:
				try:
					response = openai.ChatCompletion.create(
						model="gpt-4",
						temperature=0.75,
						messages=truncated_convo
					)
					break	# if the request was successful, break the retry loop
				except openai.error.RateLimitError:
					if retries < MAX_RETRIES - 1:	# don't sleep on the last retry
						time.sleep(RETRY_DELAY)	# wait before trying again
					retries += 1

			if retries == MAX_RETRIES:
				await ctx.send("Failed to process chunk after multiple retries due to rate limit. Please try again later.")
				continue	# move on to the next chunk

			# Extract the claims from the response and add them to the claims array
			for choice in response.choices:
				assistant_message = choice.message.content
				conversation.append({"role": "assistant", "content": assistant_message})
				summary = [(c.strip(), text_chunk) for c in assistant_message.split("\n") if c.strip()]
				infusion += summary
				await ctx.send(summary[0][0] + "\n\n")

		# for i in infusion:
		#	await ctx.send(i[0] + "\n\n")

@bot.command(aliases=['ask'])
async def iris(ctx, *, thought):
	"""
	The iris function allows certain users (testers) to directly interact with the Iris language model, providing their thoughts as input.
	It generates a response from the Iris model based on the input, sends the response as an embedded message in Discord, and provides options for users to give feedback, share the response, or request elaboration.
	The user's input and the model's response are also saved in the 'iris_training-data.csv' file for potential future training.
	"""

	global training_data, models
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden", "dpax", "Kaliyuga", "Tej", "Gregory | RND", "futurememe"]
	
	# Only Allow Some Users
	#if ctx.message.author.name not in testers:
	#	return

	thought_prompt = thought + "\n\n###\n\n"

	response = openai.Completion.create(
		model=models["chat-iris"],
		prompt=thought_prompt,
		temperature=0.08,
		max_tokens=420,
		top_p=1,
		frequency_penalty=1.5,
		presence_penalty=1.5,
		stop=["END"]
	)

	text = response['choices'][0]['text']
	text = text.replace("###", "").strip()
	embed = discord.Embed(title = "", description=f"**Prompt**\n{thought}\n\n**Response**\n{text}")

	await ctx.send(embed=embed)

	# Send Clarification and Share UI
	view, modal = response_view(modal_text="Write your clarification here", modal_label="Clarification", button_label="feedback")
	share_button = group_share(thought=text, prompter=ctx.message.author.name)
	el_prompt = thought + "\n\n" + text
	elaborate_button = elaborate(ctx, prompt=el_prompt)
	view.add_item(share_button)
	view.add_item(elaborate_button)
	await ctx.send(view=view)

	# Save Clarification
	await modal.wait()

	prompt = thought + "\n\n" + text

	if modal.answer.value is not None:
		training_data.loc[len(training_data.index)] = [prompt, modal.answer.value, ctx.message.author.name] 
		training_data.to_csv('chat-iris.csv', encoding='utf-8', index=False)

@bot.command()
async def davinci(ctx, *, thought):
	"""
	/ask query an iris and get a response
	"""

	global training_data
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	
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
	share_button = group_share(thought=text, prompt=thought_prompt, prompter=ctx.message.author.name)
	view.add_item(share_button)

	await ctx.send(embed=embed)
	await ctx.send(view=view)

@bot.command()
async def pullcard(ctx, *, intention=""):
	"""
	The pullcard function lets users draw a random tarot card with an optional intention. The drawn card, its description, and image are sent to the user. 
	If an intention is provided, the function also generates and sends an interpretation of the card in the context of the intention using the OpenAI model.
	"""

	# With Intention
	with_intention = len(intention) > 0
	r_num = random.random()
	if with_intention: random.seed(intention + str(r_num))

	# Get Embed Data
	card_name = random.choice(list(tarot_lookup.keys()))
	description = tarot_lookup[card_name].strip()

	url = ""

	try:
		for page in airtable.iterate():
			for record in page:
				if record["fields"]["Card Name"] == card_name:
					if "All images" in record["fields"]:
						url = record["fields"]["All images"][0]["url"]
	except:
		url = ""

	# Make and Send Card
	embed = discord.Embed(title = card_name, description = f"**Description**\n{description}")
	if with_intention: embed.add_field(name="Intention", value=intention, inline=False)
	if len(url) > 0: embed.set_image(url=url)
	await ctx.send(embed=embed)

	# Make and Send Card Analysis
	if with_intention:
		prompt = "My intention in this card pull is: " + intention + "\n\n"
		prompt += "You pulled the " + card_name + " card\n\n"
		prompt += description + "\n\n"
		prompt += "First explain what the intention: '" + intention + "' means, then answer how this intention connects to the card. If it's a question the intention is to know the answer. Write a few sentences and mention the intention directly. Do NOT summarize or repeat the card. Be creative in your interpretation. If the intention is one word talk more about the intention in detail"
		prompt += "\n\n"

		response = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.8,
			max_tokens=222,
			top_p=1,
			frequency_penalty=2,
			presence_penalty=2,
			stop=["END"]
		)

		text = response['choices'][0]['text'].strip()
		embed_b = discord.Embed(title = "One Interpretation (beta)", description=text)
		await ctx.send(embed=embed_b)
	
@bot.command(name="ask_group", description="Ask group a question and auto-summarize")
async def ask_group(ctx, *, question=""):
	"""
	The ask_group function lets testers ask a question to a group called "Birdies". It sends the question via direct messages, collects responses, and generates a summarized consensus using an OpenAI model. 
	The consensus is then shared with the Birdies. The function ensures a minimum of two responses for summarization and provides a redo option if needed.
	"""


	if len(question) == 0:
		return

	people = members(debug=False)
	testers = ["John Ash's Username for Discord", "JohnAsh", "EveInTheGarden"]
	users = []

	# Get Birdies
	guild = bot.get_guild(989662771329269890)
	members = discord.utils.get(guild.roles, name="Birdies").members
	birdies = [x.name for x in members]

	# Only Allow Some Users
	if ctx.message.author.name not in testers:
		return

	# Debug Mode
	for guild in bot.guilds:
		for member in guild.members:
			if member.name in birdies:
				users.append(member)

	# Get people in Garden
	responses = []
	views = []
	t_embed = discord.Embed(title = "Time Limit", description = f"Please reply within 45 minutes of receipt")
	i_url = "https://media.discordapp.net/attachments/989662771329269893/1019641048407998464/chrome_Drbki2l0Qq.png"
	c_embed = discord.Embed(title="Confluence Experiment", description = question)
	c_embed.set_image(url=i_url)

	# Message Users
	for person in users:
		view, modal = response_view(modal_text=question)
		try: 
			await person.send(embed=c_embed)
			await person.send(view=view)
			await person.send(embed=t_embed)
		except:
			continue
		responses.append(modal)
		views.append(view)

	# Gather Answers
	all_text = []

	for response in responses:
		await response.wait()
		all_text.append(response.answer.value)

	joined_answers = ""

	for t in all_text:
		if t is not None:
			joined_answers += t + "\n\n"

	if len(joined_answers.strip()) == 0:
		j_embed = discord.Embed(title = "No Responses", description = f"No responses provided to summarize")
		await ctx.send(embed=j_embed)
		return

	if len(responses) == 1:
		k_embed = discord.Embed(title = "One Response", description = all_text[0])
		await ctx.send(embed=k_embed)
		return

	# Chunk Answers
	prompts = []
	answer_chunks = textwrap.wrap(joined_answers, 1750)

	for chunk in answer_chunks:
		prompts.append(pool_prompt(question, joined_answers))

	# Query OpenAI
	response_text = ""

	for prompt in prompts:
		summarized = openai.Completion.create(
			model="text-davinci-002",
			prompt=prompt,
			temperature=0.6,
			max_tokens=222,
			top_p=1,
			frequency_penalty=1.5,
			presence_penalty=1,
			stop=["END"]
		)
		response_text += summarized.choices[0].text.strip() + "\n\n"
	
	# Send Results to People
	a_embed = discord.Embed(title = "Responses", description = f"{joined_answers}")
	embed = discord.Embed(title = "Consensus (beta)", description = f"**Question**\n{question}\n\n**Consensus**\n{response_text}")

	for person in users:
		try:
			await person.send("Responses", embed=a_embed)
			await person.send("Consensus", embed=embed)
		except:
			continue

	# Send a Redo Option
	for prompt in prompts:
		r_view = redo_view(ctx, prompt, question)
		await ctx.send(view=r_view)

bot.run(discord_key)
