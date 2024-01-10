import discord
from discord.ext import commands
import requests
import openai
from dotenv import load_dotenv
import os
import subprocess


intents = discord.Intents.default()

intents.message_content = True
client = commands.Bot(intents=intents, command_prefix="$")
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

@client.command()
async def play(ctx):
    subprocess.run(["./main.swift", ctx.author.name, ctx.message.content.replace("$play", "")], shell=True)
    await ctx.send(file=discord.File("./images/"+ ctx.author.name + ".png"))

@client.command()
async def info(ctx):
    await ctx.send("Made by Ra'Ed#2931")

@client.command()
async def users(ctx):
    await ctx.send(f"""This server has {id.member_count} members""")

client.run(TOKEN)