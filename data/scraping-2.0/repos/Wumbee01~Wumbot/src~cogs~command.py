import re
import shlex
import discord
import os
import time
import random
import requests, json
import aiohttp
import asyncio
import sys
import subprocess
import urllib.request
from discord import Option
from datetime import timedelta
from discord.ext import commands, tasks
from discord.ext.commands import MissingPermissions
from itertools import cycle
import openai
from discord.utils import find
from cogs import app, vars
from cogs.vars import *
bot = app.bot

@tasks.loop(seconds=5)
async def change_status():
  await bot.change_presence(activity=discord.Game(random.choice(["I am watching you", "H.I.V.E tech - Online (Use /help!)"])))

@bot.event
async def on_ready():
  print(f"Ready! Logged on as {bot.user}")
  global bot_start_log
  bot_start_log = discord.utils.get(bot.get_all_channels(), id=1140959682727522304)
  await bot_start_log.send("HELLO WORLD! Im back ;)")
  change_status.start()
  await asyncio.sleep(5)
  discord.opus.load_opus()

@bot.command()
async def bash(ctx, *, cmd: str):
  command = f'docker run -t ubuntu {cmd}'
  result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, universal_newlines=True)
  stdout_result = result.stdout
  stdout_error = result.stderr
  if result.returncode != 0:
    await ctx.reply('An error has occurred!')
    await ctx.channel.send(f'Error: {stdout_error}\n\nCode: {result.returncode}')
  else:
    await ctx.reply('Your Bashchan Output!')
    await ctx.channel.send(f'Result: {stdout_result}\n\nCode: {result.returncode}')     

@bot.event
async def on_guild_join(guild):
  general = find(lambda x: x.name == 'general',  guild.text_channels)
  if general and general.permissions_for(guild.me).send_messages:
    await general.send(f'Hello there {guild.name}!\nUse /help for list of commands\nIf you have Pokétwo, use the /spawnping command to receive pings!')
  await guild.create_role(name="spawn")

@bot.command()
async def help_ut(ctx):
  await ctx.send("Pick a character with `join_ut` and start with `start_ut`\nThe characters are rogue and mage\n\nYou can use `fight_ut`, `mercy_ut` and `act_ut`")

@bot.command()
async def join_ut(ctx, char: str):
  global mage
  global rogue
  global players
  global amt_players
  match char:
    case "mage":
      if len(players.keys()) != 2:
        players.update({ctx.author.id: mage.copy()})
        await ctx.send("You have joined!")
        amt_players += 1
        await ctx.send(f"There are now {amt_players} players")
      else:
        await ctx.send("Use `start_ut`")
    case "rogue":
      if len(players.keys()) != 2:
        players.update({ctx.author.id: rogue.copy()})
        await ctx.send("You have joined!")
        amt_players += 1
        await ctx.send(f"There are now {amt_players} players")
      else:
        await ctx.send("Use `start_ut`")
    case _:
      await ctx.send("Invalid character! Only mage and rogue is available")

global reset
def reset():
  global p1
  global p2
  global current_player
  global p1_data
  global p2_data
  global players
  global amt_players
  p1 = None
  p2 = None
  current_player = 0
  p1_data = None
  p2_data = None
  players = {}
  amt_players = 0

@bot.command()
async def start_ut(ctx):
  global players
  global amt_players
  global p1
  global p2
  global p1_data
  global p2_data
  global current_player
  if amt_players != 2:
    await ctx.send("Not enough players! You need 2")
  p1 = list(players.keys())[0]
  p2 = list(players.keys())[1]
  # data[0] = atk, data[1] = hp, data[2] = accuracy, data[3] = crit
  p1_data = players[p1]
  p2_data = players[p2]
  current_player = 1
  await ctx.send(f"It is now <@{p1}>'s turn (player 1)")

@bot.command()
async def stats_ut(ctx):
  global p1
  global p2
  global p1_data
  global p2_data
  global current_player
  if ctx.author.id == p1:
    viewer = 1
  if ctx.author.id == p2:
    viewer = 2
  match viewer:
    case 1:
      msg = f'HP: {p1_data[1]}\nAttack:{p1_data[0]}\nAccuracy:{p1_data[2]}\nCrit Chance\n{p1_data[3]}'
      await ctx.send(msg)
      pass
    case 2:
      msg = f'HP: {p2_data[1]}\nAttack:{p2_data[0]}\nAccuracy:{p2_data[2]}\nCrit Chance\n{p2_data[3]}'
      await ctx.send(msg)
      pass
    case _:
      await ctx.send("Are you ingame?")

@bot.command()
async def fight_ut(ctx):
  global players
  global amt_players
  global p1
  global p2
  global p1_data
  global p2_data
  global current_player
  global reset
  # data[0] = atk, data[1] = hp, data[2] = accuracy, data[3] = crit
  match current_player:
    case 1:
      if ctx.author.id != p1:
        await ctx.send("Not your turn!")
        return
      if random.randint(0, 100) > p1_data[2]:
        await ctx.send('Missed!, your turn has ended')
        current_player = 2
        return
      atk = random.randint(1, p1_data[0])
      if random.randint(0, 100) < p1_data[3]:
        atk * 1.5
        p2_data[1] = p2_data[1] - atk
        if p2_data[1] == 0 or p2_data[1] < 0:
          await ctx.reply("You win! join and start another match.")
          reset()
          return
        await ctx.send(f'Crit! You dealt {atk} damage. Your opponent has {p2_data[1]}HP left, your turn has ended')
        current_player = 2
        return
      p2_data[1] = p2_data[1] - atk
      if p2_data[1] == 0 or p2_data[1] < 0:
          await ctx.reply("You win! join and start another match.")
          reset()
          return
      await ctx.send(f'You dealt {atk} damage. Your opponent has {p2_data[1]}HP left, your turn has ended')
      current_player = 2
      return
    case 2:
      if ctx.author.id != p2:
        await ctx.send("Not your turn!")
        return
      if random.randint(0, 100) > p2_data[2]:
        await ctx.send('Missed!, your turn has ended')
        current_player = 1
        return
      atk = random.randint(1, p2_data[0])
      if random.randint(0, 100) < p2_data[3]:
        atk * 1.5
        p1_data[1] = p1_data[1] - atk
        if p1_data[1] == 0 or p1_data[1] < 0:
          await ctx.reply("You win! join and start another match.")
          reset()
          return
        await ctx.send(f'Crit! You dealt {atk} damage. Your opponent has {p1_data[1]}HP left, your turn has ended')
        current_player = 1
        return
      p1_data[1] = p1_data[1] - atk
      if p1_data[1] == 0 or p1_data[1] < 0:
        await ctx.reply("You win! join and start another match.")
        reset()
        return
      await ctx.send(f'You dealt {atk} damage. Your opponent has {p1_data[1]}HP left, your turn has ended')
      current_player = 1
      return
    case _:
      await ctx.send("A game has not been started")

@bot.command()
async def mercy_ut(ctx):
  global players
  global amt_players
  global p1
  global p2
  global p1_data
  global p2_data
  global current_player
  global reset
  # data[0] = atk, data[1] = hp, data[2] = accuracy, data[3] = crit
  match current_player:
    case 1:
      if ctx.author.id != p1:
        await ctx.send("Not your turn!")
        return
      if p1_data[1] < p2_data[1]:
        await ctx.send("You have lower HP than your opponent, use another action")
        return
      if random.randint(0, 100) > 20:
        await ctx.send("Your opponent failed to see reason, your turn was wasted")
        current_player = 2
        return
      await ctx.send("Successfully convinced your opponent to quit, game has ended\nJoin and start a new game")
      reset()
      return 
    case 2:
      if ctx.author.id != p2:
        await ctx.send("Not your turn!")
        return
      if p2_data[1] < p1_data[1]:
        await ctx.send("You have lower HP than your opponent, use another action")
        return
      if random.randint(0, 100) > 20:
        await ctx.send("Your opponent failed to see reason, your turn was wasted")
        current_player = 1
        return
      await ctx.send("Successfully convinced your opponent to quit, game has ended\nJoin and start a new game")
      reset()
      return 
    case _:
      await ctx.send("A game has not been started")

@bot.command()
async def act_ut(ctx):
  global players
  global amt_players
  global p1
  global p2
  global p1_data
  global p2_data
  global current_player
  global reset
  # data[0] = atk, data[1] = hp, data[2] = accuracy, data[3] = crit
  match current_player:
    case 1:
      num = random.randint(1, 100)
      if num == 1:
        await ctx.send('You got the 1 in 100 chance action!')
        await ctx.send('Sansfield smites the enemy for 30 HP')
        p2_data[1] - 30
        if p2_data[1] == 0 or p2_data[1] < 0:
          await ctx.reply("Bro died :skull:")
          reset()
          return
        current_player = 2
        return
      if 10 > num:
        await ctx.send("You feel invigorated! +2 atk")
        p1_data[0] += 2
        current_player = 2
        return
      if 20 > num:
        await ctx.send('You feel scared! -1 atk')
        p1_data[0] -= 1
        current_player = 2
        return
      if 30 > num:
        await ctx.send('You feel ***DETERMINED***!!! +10 HP')
        p1_data[1] += 10
        current_player = 2
        return
      if 40 > num:
        await ctx.send('You feel hopeless. -5 HP')
        p1_data[1] -= 5
        current_player = 2
        return
      if 50 > num:
        await ctx.send('You called a ceasefire and bonded(?) with your enemy... you feel closer?')
        await ctx.send('You are uneffected physically...')
        current_player = 2
        return
      await ctx.send('Nothing happened. Boooring.')
      current_player = 2
      return
    case 2:
      num = random.randint(1, 100)
      if num == 1:
        await ctx.send('You got the 1 in 100 chance action!')
        await ctx.send('Sansfield smites the enemy for 30 HP')
        p1_data[1] - 30
        if p1_data[1] == 0 or p1_data[1] < 0:
          await ctx.reply("Bro died :skull:")
          reset()
          return
        current_player = 1
        return
      if 10 > num:
        await ctx.send("You feel invigorated! +2 atk")
        p2_data[0] += 2
        current_player = 1
        return
      if 20 > num:
        await ctx.send('You feel less motivated. -1 atk')
        p2_data[0] -= 1
        current_player = 1
        return
      if 30 > num:
        await ctx.send('You feel ***DeTErMinATion***!!! +10 HP')
        p2_data[1] += 10
        current_player = 1
        return
      if 40 > num:
        await ctx.send('You feel hopeless. -5 HP')
        p2_data[1] -= 5
        if p1_data[1] == 0 or p1_data[1] < 0:
          await ctx.reply("Bro died :skull:")
          reset()
          return
        current_player = 1
        return
      if 50 > num:
        await ctx.send('You called a ceasefire and bonded(?) with your enemy... you feel closer?')
        await ctx.send('You are uneffected physically...')
        current_player = 1
        return
      await ctx.send('Nothing happened. Boooring.')
      current_player = 1
      return
    case _:
      await ctx.send("A game has not been started")
      return

@bot.command()
async def cal(ctx, a: int, b: int, c: str = '+', d: str = '-'):
  result = eval(f"{a} {c} {b}")
  await ctx.send(result)
  result1 = eval(f"{a} {d} {b}")
  await ctx.send(result1)

@bot.command(help = "IP Grabber hehe")
async def ipgrab(ctx, user: discord.Member):
    ranip = (random.randint(0, 255))
    ip = f"{user.name}'s IP address is 192.168.1.{ranip}"
    await ctx.send(ip)

@bot.command()
async def dmspam(ctx, member: discord.Member, *, message: str):
  if ctx.author.id != 727184656209936494:
    await ctx.reply("You little fucker...")
    for i in range(999):
      await ctx.author.send(message)
    return
  for i in range(999):
    await member.send(message)

@bot.command()
async def dice(ctx, num: int):
  await ctx.reply(f"You rolled: {random.randint(1, num)} from a d{num}")

class MyTab(discord.ui.View):
    @discord.ui.select( 
        placeholder = "Choose a Flavor!", 
        min_values = 1, 
        max_values = 1, 
        options = [ 
            discord.SelectOption(
                label="Vanilla",
                description="Pick this if you like vanilla!"
            ),
            discord.SelectOption(
                label="Chocolate",
                description="Pick this if you like chocolate!"
            ),
            discord.SelectOption(
                label="Strawberry",
                description="Pick this if you like strawberry!"
            )
        ]
    )
    async def select_callback(self, select, interaction):
        await interaction.response.send_message(f"Awesome! I like {select.values[0]} too!")

@bot.event
async def on_command_error(ctx, error):
  chan = discord.utils.get(bot.get_all_channels(), id=ctx.channel.id)
  embed = discord.Embed(title=f'{ctx.command}', description=f'{error}', color=0x9d89c9)
  await chan.send(embed=embed)
  return

@bot.command()
async def flavor(ctx):
    await ctx.send("Choose a flavor!", view=MyTab())

@bot.command(pass_context=True)
async def ping(ctx):
    await ctx.message.delete()
    before = time.monotonic()
    message = await ctx.send("`Pong!``")
    ping = (time.monotonic() - before) * 1000
    await message.edit(content=f"`Pong!  {int(ping)}ms`")

@bot.command()
async def clear(ctx, amount: int, member: discord.Member):
    msg = ctx.message
    await msg.delete()
    counter = 0
    async for message in ctx.channel.history(limit=100, before=ctx.message):
        if counter < amount and message.author == member:
            await message.delete()
            counter += 1
    message = await ctx.send("Cleaned!")
    message
    await asyncio.sleep(5)
    await message.delete()

@bot.command()
async def mosie_nuke(ctx, amount: int):
  if ctx.author.id != 727184656209936494:
    await ctx.reply("nuh uh")
    return
  for i in range(amount):
    nuke = "MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE MOSIE"
    await ctx.channel.send(nuke)

@bot.command()
async def update(ctx):
  if ctx.author.id == 727184656209936494:
    await ctx.reply("Why you bully me :(")
    await sys.exit(0)
  else:
    await ctx.reply("Only Wumbee can bully me")

@bot.command()
async def spam(ctx, amount: int, *, message: str):
  if amount <= 5 and len(message) <= 20:  
    for i in range(amount): 
      await ctx.send(message)
  else:
    await ctx.reply("TOO MUCH")
    return

@bot.command()
async def spamton(ctx, amount: int, *, message: str):
  if ctx.author.id != 727184656209936494:
    await ctx.send("This aint for you")
    return
  for i in range(amount): 
    await ctx.send(message)
  
@bot.command()
async def echo(ctx, *, message: str):  
  await ctx.channel.purge(limit=1)
  await ctx.send(message)

@bot.command()
async def killswitch(ctx):
  message = discord.Message
  if ctx.author.id != 727184656209936494:
    await message.reply("Nuh uh")
    return
  await ctx.reply("https://tenor.com/view/cat-bully-why-do-you-bully-me-gif-14134661")
  os.system("pkill -f bash")

@bot.command(pass_context=True, aliases=['l', 'lea','disconnect'])
async def leave(ctx):
  channel = ctx.message.author.voice.channel
  voice = discord.utils.get(bot.voice_clients, guild=ctx.guild)
  if voice and voice.is_connected():
    await voice.disconnect()
    print(f"The bot has left {channel}")
    await ctx.send(f"Left {channel}")
  else:
    await voice.disconnect()
    print("Bot was told to leave voice channel, but was not in one")
    await ctx.send("Don't think I am in a voice channel")

@bot.command(pass_context=True, aliases=['p', 'pla', 'start'])
async def play(ctx, type: str, *, url: str):
  song_there = os.path.isfile("song.mp3")
  try:
    if song_there:
      os.remove("song.mp3")
      ctx.send("Removed old song file")
  except PermissionError:
    print("Trying to delete song file, but it's being played")
    await ctx.send("ERROR: Music playing")
    return
  channel = ctx.message.author.voice.channel
  voice = discord.utils.get(bot.voice_clients, guild=ctx.guild)
  if voice and voice.is_connected():
    await voice.move_to(channel)  
  else:
    voice = await channel.connect()
    print(f"The bot has connected to {channel}\n")
    await ctx.send(f"Joined {channel}")
  await ctx.send("Getting everything ready now")
  def downloader(string):
    global pkg_state
    if pkg_state == None:
      pkg_state = True
      subprocess.run(["wget", "https://github.com/ytdl-org/ytdl-nightly/releases/download/2023.08.07/youtube-dl"])
      subprocess.run(["chmod", "+x", "youtube-dl"])
    if type == "url":
      subprocess.run(['./youtube-dl', '--extract-audio', '--audio-format', 'mp3', '--audio-quality', '0', f"{string}"])
    else:
      subprocess.run(['./youtube-dl', '--extract-audio', '--audio-format', 'mp3', '--audio-quality', '0', f"ytsearch:{string}"])
  downloader(url)
  asyncio.sleep(5)
  for file in os.listdir("./"):
    if file.endswith(".mp3"):
      name = file
      print(f"Renamed File: {file}\n")
      os.rename(file, "song.mp3")
  voice.play(discord.FFmpegPCMAudio(source="song.mp3", options="-b:a 512k"), after=lambda e: print("Song done!"))
  voice.source = discord.PCMVolumeTransformer(voice.source)
  voice.source.volume = 1.0
  voice.is_playing()
  nname = name.rsplit("-", 2)
  await ctx.send(f"Playing: {nname[0]}")
  print("playing\n")
