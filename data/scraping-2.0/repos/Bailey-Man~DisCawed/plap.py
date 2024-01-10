# dnd discord bot implementation
import regex as re
import os 
import sys 
import datetime
import discord
from discord.ext import commands
import random 
import requests
if sys.path.count('src') == 0:
    sys.path.append('src') 
else:
    print('src already in path')
    print('sys.path', sys.path) # get rid of this when tested

from src.commands import helper



# from openai import OpenAI

from dotenv import load_dotenv

## init
load_dotenv() # do i do this in script or within the bot init??

# force intents
myintents = discord.Intents.default()
myintents.members = True
myintents.message_content = True

bot = commands.Bot(command_prefix='!', intents=myintents)


## EVENTS ##
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.errors.CheckFailure):
        await ctx.send('you do not have the correct role for this command')


### TEST THESE ###
# when player accepts invite to the server, send a message to admin channel with who joined and who invited them
@bot.event
async def on_member_join(member):
    # get the admin channel
    admin_channel = discord.utils.get(member.guild.channels, name='admin')
    # get the invite that was used to join the server
    invite = await admin_channel.create_invite(max_age=300)
    # get the user who created the invite
    inviter = invite.inviter
    # send a message to the admin channel with who joined and who invited them
    await admin_channel.send(f'{member.name} joined using invite created by {inviter.name}')

# when player joins the server, check if they are in the database.
# if they are, populate their character sheet with that data
# if they are not, ask them to run the !register command
# @bot.event
# async def on_member_join(member):
#     # if member in preset list; populate with that
#     # else; diagnose from avatar and username
#     if member in preset_list:
#         pass

# whenever a message contains the substring 'BUDDY', respond with 'BUDDYY' with 3 times as many Y's as the original message
@bot.event
async def on_message(message):
    if 'BUDDY' in message.content:
        response = 'BUDDY' + 'Y' * (len(message.content) - 5) 
        await message.channel.send(response)
    await bot.process_commands(message)





## COMMANDS ##
@bot.diceroll(name='roll', help='roll dice')
async def roll(ctx, *args):
    # take a string that contains a dice roll in format NdN with an optional 'w adv' or 'w disadv'
    input_string = ' '.join(args)
    if input_string == '':
        await ctx.send('please provide a dice roll in the format NdN')
        return None
    
    # regex to extract whether the dice roll has advantage or disadvantage
    adv_pattern = r'(w adv|w disadv)'
    dice_pattern = r'(\d+)d(\d+)'
    adv_match = re.search(adv_pattern, input_string) # if adv_pattern matches    
    dice_match = re.search(dice_pattern, input_string)
    adv_status = None
    if adv_match:
        # determine if adv or disadv
        adv_or_disadv = adv_match.group(1) # ? what does this do ??
        print('adv_or_disadv', adv_or_disadv)
        if adv_or_disadv == 'w adv':
            adv_status = True
        elif adv_or_disadv == 'w disadv':
            adv_status = False
        else:
            print('adv_match', adv_match, adv_match.group(1))
            await ctx.send('please provide a dice roll in the format NdN')
    
    if dice_match == None:
        await ctx.send('please provide a dice roll in the format NdN')
        return None
    else:
        number_of_dice = int(dice_match.group(1))
        number_of_sides = int(dice_match.group(2))
        # roll the dice
        myroll = helper.roll_dnd_dice(adv_status, number_of_dice, number_of_sides)
        myresponse = f'your roll of {number_of_dice}d{number_of_sides} is {myroll}'

        await ctx.send(myresponse)



# ban a randomly selected user from the server
@bot.command(name='random_ban', help='ban a randomly selected user from the server')
async def random_ban(ctx):
    # get the list of members in the server
    member_list = ctx.guild.members

    # randomly select a member and assert its not the user that called the command
    member = random.choice(member_list)
    while member == ctx.author:
        member = random.choice(member_list)
    # ban the member    
    await member.ban()
    # send a message to the channel that the member was banned
    await ctx.send(f'{member.name} was banned')

#




