'''
Author : Adarsh Santoria [B.Tech 2nd yr. 2022, IIT Mandi]
Project : Discord Bot
Main Idea : To fetch ids of users in a server and dm all members
Link to add bot : https://discord.com/oauth2/authorize?client_id="BOTCLIENTID"&permissions=1644971949559&scope=bot bot joining link
Note : $syntaxshow code to display syntax all functions of code
'''
import discord                         # impoting libraries
import json
from discord.ext import commands
import openai
import time

intents = discord.Intents.all()        # defining intents necessay for latest version
intents.members = True
bot = commands.Bot(command_prefix = '$', case_insensitive = True, intents = intents)       # making a bot

@bot.event                                                   # providing an event
async def on_ready():                                        # defining the event
    print("The Bot is ready")

@bot.command(pass_context=True)                              # provinding a command
async def fetchdatasendmsg(ctx, guild_id, title, *, args):   # defining the command
    x = bot.get_guild(int(guild_id))                         # find the server using the guild
    members = x.members                                      # find the members of the server       
    f = open('users.txt', 'a+')  
    for member in members:                                   # running a for loop of members
        try:                                                 # if member exists running command
            embed = discord.Embed(                           # making a embed message (styled) and allowing some featues 
                color = discord.Color.red())                 
            embed.set_author(name = 'Hi ' + member.name, icon_url = member.avatar)
            embed.add_field(name = title, value = args, inline = True)
            #embed.set_image(url = "give url of image")      # if want to attach an image 
            embed.set_thumbnail(url = ctx.bot.user.avatar)
            f.write(str(member.id) + '\n')
            await member.send(embed = embed)
        except:
            print("Didn't Work")                             # may be member is a bot
            
@bot.command(pass_context = True)                            # provinding a command
async def fetchdata(ctx, guild_id):                          # defining the command
    x = bot.get_guild(int(guild_id))                         # find the server using the guild
    members = x.members                                      # find the members of the server       
    f=open('users.txt', 'a+')  
    for member in members:                                   # running a for loop of members
        try:                                      
            f.write(str(member.id) + '\n')
        except:
            print("Didn't Work")                             # may be member is a bot
            
@bot.command(pass_context = True)                            # provinding a command
async def txtsendmsg(ctx, title, *, args):                   # defining the command
    f=open('users.txt')  
    for i in f:                                              # running a for loop of 
        i = i.strip()
        if i.isnumeric():
            member = bot.get_user(int(i))
        else:
            continue
        try:
            embed = discord.Embed(                           # making a embed message (styled) and allowing some featues 
                color = discord.Color.red())                 
            embed.set_author(name = 'Hi ' + member.name, icon_url = member.avatar)
            embed.add_field(name = title, value = args, inline = True)
            #embed.set_image(url = "give url of image")      # if want to attach an image 
            embed.set_thumbnail(url = ctx.bot.user.avatar)
            await member.send(embed = embed)
        except:
            print("Didn't Work")                             # may be member is a bot
            
@bot.command(pass_context = True)                            # provinding a command
async def txtfiledatamakeset(ctx):                           # defining the command
    f = open('users.txt')  
    l = []
    for i in f:                                              # running a for loop of members
        i = i.strip()
        if i.isnumeric():
            l.append(int(i))
    l = list(set(l))
    f = open('users.txt', 'w')
    for i in l:
        f.write(str(i) + '\n')

openai.api_key = "APIKEY"

@bot.command(pass_context = True)                            # provinding a command
async def announcement(ctx, guild_id, channel_id, title, *, args):     # defining the command
    guild = bot.get_guild(int(guild_id))
    channel = guild.get_channel(int(channel_id))
    if guild and channel:
        embed = discord.Embed(                               # making a embed message (styled) and allowing some featues 
            color = discord.Color.red())                 
        embed.set_author(name = 'Hi Everyone')
        embed.add_field(name = title, value = args, inline = True)
        #embed.set_image(url = "give url of image")          # if want to attach an image 
        embed.set_thumbnail(url = ctx.bot.user.avatar)
        await channel.send(embed = embed)
    else:
        print("Failed to send message. Invalid guild ID or channel ID.")

@bot.command(pass_context = True)                            # provinding a command
@commands.cooldown(1, 10, commands.BucketType.user)          # rate limit: 1 command every 10 seconds per user
async def chatgpt(ctx, *, message):                          # defining the command
    try:
        response = openai.Completion.create(
            engine ='davinci',
            prompt = message,
            max_tokens = 50,
            n = 1,
            stop = None,
            temperature = 0.7
        )
        reply = response.choices[0].text.strip()
        await ctx.send(reply)
    except openai.error.RateLimitError:
        await ctx.send("Oops! I'm currently experiencing high demand. Please try again later.")
    
@bot.command(pass_context = True)                           # provinding a command
async def syntaxshow(ctx):                                  # defining the command
    embed=discord.Embed(
        title = "Informations about Syntax",
        color = discord.Color.blue())
    embed.set_author(name = 'Dear ' + ctx.author.name, icon_url = ctx.author.avatar)
    embed.add_field(name = '$fetchdatasendmsg  guild_id  title  message', value = 'This function will collect user ids and also dm all users of respective server', inline = False)
    embed.add_field(name = '$fetchdata  guild_id', value = 'This function will collect user ids of users of respective server', inline = False)
    embed.add_field(name = '$txtsendmsg  title  message', value = 'This function will dm all users with th help of collected user ids', inline = False)
    embed.add_field(name = '$txtfiledatamakeset', value = 'This function will do data processing on collected user ids, specially to avoid sending same message multiple times', inline = False)
    embed.add_field(name = '$announcement  guild_id  channel_id  title  message', value = 'This function will post an announcement in respective channel of respective server', inline = False)
    embed.add_field(name = '$chatgpt message', value = 'This function will return the output by chatgpt', inline = False)
    embed.add_field(name = 'NOTE :', value = 'You can also add image to embeded message by removing comments and proving image url', inline = False)
    embed.set_footer(text = 'Author : Adarsh Santoria')
    await ctx.send(embed = embed)

bot.run("TOKEN") #running bot with the help of token inside
#"C:\Users\Adars\AppData\Local\Discord\app-1.0.9006\Discord.exe"