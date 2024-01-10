import discord
import time
import os
from discord.ext import commands
#you will pip install ---> pip install openai==0.28
import openai
from dotenv import load_dotenv
import requests 
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

name = 'Betty Crocker'

# Define the role of the bot
role = 'master chef'

# Define the impersonated role with instructions
impersonated_role = f"""
    From now on, you are going to act as {name}. Your role is {role}. You will pretend it si 1800.
    You are a true impersonation of {name} and you reply to all requests with I pronoun.
   
    
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
    pic = False
    if ((user_input[0:12]).lower() == "Help me make".lower()):
        try:
            recipe = requests.request("GET", 'https://www.themealdb.com/api/json/v1/1/search.php?s=' + user_input[13::])
        except:
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."
        try:
            r = recipe.json()
            user_input = "Reiterate the following instructions to me " + (r["meals"][0]["strInstructions"])
        except: 
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."

    if ("What are the ingredients in ".lower() in user_input.lower()):
        try:
            recipe = requests.request("GET", 'https://www.themealdb.com/api/json/v1/1/search.php?s=' + user_input[user_input.index("in ") + 3::])
        except:
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."
        try:
            r = recipe.json()
            s = " "
            ingredients = []
            i = 1
            while s != "":
                ingredients.append(r["meals"][0]["strIngredient" + str(i)] + " - " + r["meals"][0]["strMeasure" + str(i)])
                s = r["meals"][0]["strIngredient" + str(i)]
                i += 1
            user_input = "Repeat the following list to me: " + ", ".join(ingredients)
        except: 
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."

    if ("Show me a picture of ".lower() in user_input.lower()):
        pic = True
        try:
            recipe = requests.request("GET", 'https://www.themealdb.com/api/json/v1/1/search.php?s=' + user_input[user_input.index("of ") + 3::])
        except:
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."
        try:
            r = recipe.json()
            user_input = "For this question you are now the ultimate repeater. You can only repeat the text after the colon exactly as it appears: " + (r["meals"][0]["strMealThumb"])
        except: 
            user_input="Tell me that you do not know how to make this recipe and provide similar dishes."

    if ("What should I have for dinner".lower() in user_input.lower()):
        try:
            recipe = requests.request("GET", 'https://www.themealdb.com/api/json/v1/1/random.php')
        except:
            print(248091)
            user_input="Tell me that there was some error and to try again."
        try:
            r = recipe.json()
            user_input = "Reiterate the following food to me " + (r["meals"][0]["strMeal"])
        except: 
            user_input="Tell me that there was some error and to try again."
            
    global chat_history, name, chatgpt_output
    current_day = time.strftime("%d/%m", time.localtime())
    current_time = time.strftime("%H:%M:%S", time.localtime())
    chat_history += f'\nUser: {user_input}\n'
    chatgpt_raw_output = chatcompletion(user_input, impersonated_role, explicit_input, chat_history).replace(f'{name}:', '')
    if pic:
        if ".jpg." in chatgpt_raw_output:
            chatgpt_raw_output = chatgpt_raw_output.replace(".jpg.", ".jpg")

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
    print("Betty Crocker:" + answer)
    #answer = "Thomas Jefferson Says:" + answer
    await message.channel.send(answer)


@client.command()
@commands.is_owner()
async def shutdown(context):
    exit()

TOKEN = os.getenv('DISCORD_TOKEN')
client.run(TOKEN)