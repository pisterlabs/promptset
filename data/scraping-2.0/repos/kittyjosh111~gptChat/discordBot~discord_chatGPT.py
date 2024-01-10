import openai
import os
import json
import discord
from discord.ext import commands

openai.api_key = ""
discord_bot_token = ''
triggers = ["hey gpt"] #This is how to activate the bot. The bot will respond when it detects this word in any message sent on a server.

#This is a simple script to converse with OpenAI's GPT models. It tries to keep persistence between chats by creating a file to store logs of the past conversations, here known as neuralcloudv2.ncb. 
#Model responses are also written to a log.log for further reference.
#This script uses the chat model, or currently the gpt-3.5 model that is similar to ChatGPT.
#This script also will take output and inputs by interfacing with discord. Fun!

#################
### Variables ###

#counter variable that determines whether to begin with the model or the user
counter = 0

#model is the used OpenAI model. Check their website for different model names.
#https://platform.openai.com/docs/models/overview
model="gpt-3.5=turbo"

#the prompt is what the model will read for to create the response.
#Do not include the initial human prompt, just the description of what the model's pesonality should be like.
base_prompt="""The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."""

#################
#################

#First, a function to save the memory variable to the ncb. I will use this a lot, so it works best as a function.
def save_ncb():
  with open('neuralcloud_discord.ncb', 'w') as save:
     save.write(json.dumps(memory)) 

#Initialize my custom memory file. Basically, a text file to log everything we've written and then reuse it as the prompt for future prompts. 
#First we check if there already exists a neural cloud file. If not, then we create the ncb file and wrtie the prompt to it.
#Its Like waking up their neural cloud for the first time. Otherwise, its just restoring their neural clouds.
memory=[] #unlike the gpt3 script, we use a variable to store memory here. 
ncb = './neuralcloud_discord.ncb'
check = os.path.isfile(ncb)
if check:
  with open('neuralcloud_discord.ncb') as read:
    output = read.read()
  formatted_list = json.loads(output)
  memory = formatted_list #These steps allow the model to have past dialogues loaded as a python list
else:
  memory.append({"role": "system", "content": f"{base_prompt}"}, ) #creating the file with the system prompt
  memory.append({"role": "user", "content": "Hello."}, )
  save_ncb() #So the model's first words are a greeting to the user.
  counter = 1 #now the model goes first.

#################
### Functions ###

#Function for the api request so that I don't have to copy paste this over and over again.
def api_request(prompt):
  response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=prompt
  )
  api_request.response = response['choices'][0]['message']['content'].strip() 
  memory.append({"role": "assistant", "content": f"{api_request.response}"}, ) #write to the memory variable
  save_ncb() #save memory to ncb after generation of response
  log = open("logv2.log", "a")
  log.write("\n" + api_request.response) #Write to log
  log.close()

#################################################################################
# This is the discord bot portion.

intents = discord.Intents().all()
client = commands.Bot(command_prefix=',', intents=intents)

@client.event
async def on_ready():
    print('online')
    print(memory)

@client.event 
async def on_message(message):
    if message.author == client.user:
        return
    if message.author.bot: return
    for i in range(len(triggers)):
        if triggers[i].lower() in message.content.lower():
            memory.append({"role": "user", "content": message.content}, )
            save_ncb()
            api_request(memory)
            print(api_request.response)
            await message.channel.send(api_request.response)

client.run(discord_bot_token)