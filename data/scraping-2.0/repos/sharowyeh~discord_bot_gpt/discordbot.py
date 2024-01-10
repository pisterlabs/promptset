import os
import sys
import logging
import asyncio
import discord
# py-cord must upgrade to v2.x for discord option class (python v3.8)
from discord import option
from discord.ext import commands
# use gptchat from myself's twitch chat bot submodule, because
# I dont want to maintain openai chat completion behavior twice
from twitch_bot_gpt.gptchat import GPTChat
#import openai

logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'))
logger.addHandler(handler)

if os.environ.get('DISCORD_BOT_TOKEN') is None:
    print('bot token can not be empty')
    exit()

if os.environ.get('OPENAI_API_KEY') is None:
    print('openai api key can not be empty')
    exit()

#openai.api_key = os.environ.get('OPENAI_API_KEY')

intents = discord.Intents.all()

bot = commands.Bot(command_prefix='/', intents=intents)
gptchat = GPTChat()
#TODO: make some initial assistant context
#gptchat.setInitAssistant('')

@bot.event
async def on_ready():
    logger.debug(f'bot is ready, user:{bot.user}')

@bot.event
async def on_message(message: discord.Message):
    logger.debug(f'on_message, author:{message.author} msg:{message.content}')
#    if message.author == bot.user:
#        logger.debug('on bot message something')

@bot.slash_command(name='echo', description='Just make sure bot is still alive')
@option('message', description='The message')
async def echo(ctx, message: str):
    logger.debug(f'echo msg:{message}')
    await ctx.respond(message)

@bot.slash_command(name='hello', description='Just let bot say hello world')
@option('text', description='Additional text')
async def hello(ctx, text: str=''):
    logger.debug(f'{ctx.author} hello world! {text}')
    await ctx.respond(f'{ctx.author} hello world! {text}')

@bot.slash_command(name='chat', description='Talk to GPT3')
@option('text', description='The text message')
@option('temp', description='Completion temperature')
async def chat(ctx, text: str, temp: float=0.5):
    """Talk to GPT3,
    :param text: The text message
    :param temp: Completion temperature
    """
    if temp < 0 or temp > 1 or round(temp, 1) != temp:
        temp = 0.5
    logger.debug(f'{ctx.author} text:{text} temp:{temp}')

    # reply message first for context prevent task terminated before openai response(I guess?)
    reply_text = f"{ctx.author} said '{text}'"
    msg = await ctx.respond(f"{reply_text}")

    # let openai API call by async? but completion does not have acreate method
    # or use from asgiref.sync import sync_to_async? [link](https://github.com/openai/openai-python/issues/98)
    token_length = 100
    gptchat.setTemp(temp)

    reply_text += "\n" + gptchat.chatCompletion(text, token_length)
    reply_text += "\n🤔"
    await msg.edit_original_response(content=f"{reply_text}")
    print("==== end of resp ====")

bot.run(os.environ.get('DISCORD_BOT_TOKEN'))
exit()
