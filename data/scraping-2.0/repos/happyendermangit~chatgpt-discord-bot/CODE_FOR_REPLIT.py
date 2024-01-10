import discord
import openai 
import os
from discord.ext import commands 
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='',intents=intents)
token = os.getenv("TOKEN") # Discord Bot Token 

channel_id = 0 # Channel ID 
msgs = [{"role":"system",'content':'You are a helpful assistant based on gpt-3.5-turbo. Your name is ChatGPT.'}]
def chat(c):
    global msgs 
    openai.api_key = os.getenv("OPENAI_TOKEN") # OpenAI API Key
    msgs.append({"role":"user","content":c})
    cbot = openai.ChatCompletion.create(model='gpt-3.5-turbo',messages=msgs)
    msgs.append(cbot['choices'][0]['message'])
    return cbot['choices'][0]['message']['content']
@client.event 
async def on_ready():
    print("ready!",client.user)
@client.event 
async def on_message(msg):
    if msg.author.bot:
        return 
    else:
        if msg.channel.id == channel_id:
            a = await msg.channel.send('**Please wait...**')
            res = chat(msg.content)
            await a.edit(content='',embed=discord.Embed(title='GPT:',description=f'**This uses OpenAI API.**\n> {msg.content}\n{res}'))
client.run(token)
