import discord
from discord.ext import tasks, commands
import nest_asyncio
import os
import random
import time
from os.path import exists
import subprocess
from time import sleep
import datetime
import asyncio
import openai
import json
from lxml import html
from lxml import etree
import requests
import threading
from threading import Thread
import cogs.themes
from pytz import timezone

#############################################################################################
######################          Global Initializations          #############################
#############################################################################################

intents = discord.Intents.all()
nest_asyncio.apply()
bot = commands.Bot(command_prefix='$', intents=intents)
client = discord.Client(intents=intents)
blank = "‎"
defaultEmbedColor=discord.Color(0xe67e22)
green = discord.Color(0x00FF00)
red = discord.Color(0xFF0000)

initial_extensions = ['cogs.help',
                      'cogs.themes',
                      'cogs.birthday',
                      'cogs.role',
                      'cogs.dev',
                      'cogs.ai',
                      'cogs.service',
                      #'cogs.tiktok', #Retired due to being IP banned.
                      'cogs.riot', #Retired due to incompleteness/underuse of features
                      'cogs.events',
                      'cogs.qrcode',
                      'cogs.fun',
                      'cogs.moderation',
                      'cogs.vc',
                      'cogs.gifs',
                      #'cogs.warframe', #Retired due to no solid API being available
                      'cogs.gm',
                      'cogs.dm',
                      'cogs.lyrics',
                      'cogs.tasks',
                      #'cogs.trickrtreat', #Retired until October 2024
                      'cogs.youtube']

###########################################################

###########################################################

@bot.command(name="enable", hidden=True)
@commands.has_permissions(administrator = True)
async def enable(ctx, arg=None):
    if arg == None:
        errorEmbed = discord.Embed(color=red)
        errorEmbed.add_field(name=":x: Please specify a cog to add!",value="(use $cogs to view them all)")
        await ctx.reply(embed=errorEmbed)
    try:
        await bot.load_extension("cogs."+arg)
        successEmbed = discord.Embed(color=green)
        successEmbed.add_field(name=f":white_check_mark: {arg} Cog enabled!", value="Enjoy the functionality!")
        await ctx.reply(embed=successEmbed)
    except:
        errorEmbed = discord.Embed(color=red)
        errorEmbed.add_field(name=":x: That is not a valid Cog!",value="(use $cogs to view them all)")
        await ctx.reply(embed=errorEmbed)

@bot.command(name="disable", hidden=True)
@commands.has_permissions(administrator = True)
async def diable(ctx, arg=None):
    if arg == None:
        errorEmbed = discord.Embed(color=red)
        errorEmbed.add_field(name=":x: Please specify a cog to remove!",value="(use $cogs to view them all)")
        await ctx.reply(embed=errorEmbed)
    try:
        await bot.unload_extension("cogs."+arg)
        successEmbed = discord.Embed(color=green)
        successEmbed.add_field(name=f":white_check_mark: {arg} Cog disabled!", value="Hold tight while we conduct maintenance")
        await ctx.reply(embed=successEmbed)
    except:
        errorEmbed = discord.Embed(color=red)
        errorEmbed.add_field(name=":x: That is not a valid Cog!",value="(use $cogs to view them all)")
        await ctx.reply(embed=errorEmbed)

@bot.command(name="cogs", hidden=True)
@commands.has_permissions(administrator = True)
async def cogs(ctx):
    value = ""
    for ext in initial_extensions:
        name = str(ext).replace("cogs.","")
        value += name+'\n'
    cogEmbed = discord.Embed(color=defaultEmbedColor)
    cogEmbed.add_field(name="List of Cogs:", value=value)
    await ctx.reply(embed=cogEmbed)

###########################################################

async def loadall():
    for ext in initial_extensions:
        await bot.load_extension(ext)

bot.remove_command('help')
asyncio.run(loadall())

@bot.event
async def on_ready():
    print('Logged on as {0}!'.format(bot.user))
    print("Discord.py Version: "+discord.__version__)
    themes = bot.get_cog("ThemesCog")
    await themes.changeStatus()

bot.run(os.getenv("DPY_key"))
