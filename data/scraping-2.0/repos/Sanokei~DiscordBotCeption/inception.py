import os
import json
import asyncio

import openai

import discord
from discord.ext import commands

bot = commands.Bot(command_prefix='!')

with open('config.json') as config_file:
    config = json.load(config_file)

openai.api_key = config['api_key']
TOKEN = config['token']

@bot.event
async def on_ready():
    print('Logged in as')
    print(bot.user.name)
    print(bot.user.id)
    print('------')

@bot.command(pass_context=True)
async def createBot(ctx):
    def check(message):
        return message.channel == ctx.channel and message.author != ctx.me
    await ctx.reply('Create a discord bot that:')
    msg = await bot.wait_for('message',check=check)
    text = msg.content
    response = openai.Completion.create(
        engine="code-davinci-001",
        prompt="\"\"\"\nUsing Discord.py\n(Making sure to abstract the Token)\nCreate a bot that:\n"+text+"\n\n\"\"\"",
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    with open('Bad_Discord_Bot.py', 'w') as f:
        f.write("'''\n"+text+"\n'''" + response.choices[0].text)
    await ctx.send(file=discord.File('Bad_Discord_Bot.py'))
    os.remove('Bad_Discord_Bot.py')

bot.run(TOKEN)