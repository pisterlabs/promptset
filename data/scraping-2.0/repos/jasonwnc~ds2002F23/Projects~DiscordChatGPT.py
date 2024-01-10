import discord
import time
import os
from discord.ext import commands
#you will pip install ---> pip install openai==0.28
import openai
from dotenv import load_dotenv 
load_dotenv()
#You  pip install python-dotenv
# Initialize variables for chat history
explicit_input = ""
chatgpt_output = 'Chat log: /n'
cwd = os.getcwd()
i = 1

# Find an available chat history file
while os.path.exists(os.path.join(cwd, f'chat_history{i}.txt')):
    i += 1

history_file = os.path.join(cwd, f'chat_history{i}.txt')

# Create a new chat history file
with open(history_file, 'w') as f:
    f.write('\n')

# Initialize chat history
chat_history = ''

#OPEN AI STUFF
#Put your key in the .env File and grab it here
openai.api_key = os.getenv("OPENAI_API_KEY")

name = 'Thomas Jefferson'

# Define the role of the bot
role = 'customer service'

# Define the impersonated role with instructions
impersonated_role = f"""
    From now on, you are going to act as {name}. Your role is {role}. You will pretend it si 1800.
    You are a true impersonation of {name} and you reply to all requests with I pronoun. You will speak like a mid 17th century colonist in America.
   
    
"""
# Function to complete chat input using OpenAI's GPT-3.5 Turbo
def chatcompletion(user_input, impersonated_role, explicit_input, chat_history):
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        temperature=1,
        presence_penalty=0,
        frequency_penalty=0,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": f"{impersonated_role}. Conversation history: {chat_history}"},
            {"role": "user", "content": f"{user_input}. {explicit_input}"},
        ]
    )

    for item in output['choices']:
        chatgpt_output = item['message']['content']

    return chatgpt_output

# Function to handle user chat input
def chat(user_input):
    global chat_history, name, chatgpt_output
    current_day = time.strftime("%d/%m", time.localtime())
    current_time = time.strftime("%H:%M:%S", time.localtime())
    chat_history += f'\nUser: {user_input}\n'
    chatgpt_raw_output = chatcompletion(user_input, impersonated_role, explicit_input, chat_history).replace(f'{name}:', '')
    chatgpt_output = f'{name}: {chatgpt_raw_output}'
    chat_history += chatgpt_output + '\n'
    with open(history_file, 'a') as f:
        f.write('\n'+ current_day+ ' '+ current_time+ ' User: ' +user_input +' \n' + current_day+ ' ' + current_time+  ' ' +  chatgpt_output + '\n')
        f.close()
    return chatgpt_raw_output



#DISCORD STUFF

intents = discord.Intents().all()
client = commands.Bot(command_prefix="!", intents=intents)


@client.event
async def on_ready():
    print("Bot is ready")

@client.command()
async def hi(ctx):
    await ctx.send("Hello, nothing to see here")


responses = 0
list_user = []


@client.event
async def on_message(message):
    print(message.content)
    if message.author == client.user:
        return    
    print(message.author)
    print(client.user)
    print(message.content)
    answer = chat(message.content)
    print("Thomas Jefferson Says:" + answer)
    #answer = "Thomas Jefferson Says:" + answer
    await message.channel.send(answer)


@client.command()
@commands.is_owner()
async def shutdown(context):
    exit()

TOKEN = os.getenv('DISCORD_TOKEN')
client.run(TOKEN)