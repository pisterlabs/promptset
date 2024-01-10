import os

import discord
import openai
import functools
import typing
import asyncio
from dotenv import load_dotenv
import hashlib
import random

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
GUILD = os.getenv('DISCORD_GUILD')
SECURITY_PROMPT = os.getenv('SECURITY_PROMPT')

openai.api_key = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)

def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper

def init_intent():
    return {
        'chat_gpt' : False,
        'model' : 'gpt-3.5-turbo',
        'chat_msgs' : [],
        'command_txt' : {'role' : 'system', 'content' : SECURITY_PROMPT},
        'current_token' : 0,
        'key' : '',
        'users' : [],
        'token_limit' : 2000
    }

secret = 'abbb'

def format_msg(role: str, msg: str):
    return {"role": role, "content": msg}

@to_thread
def generate(model, messages):
    return openai.ChatCompletion.create(model=model, messages=messages)

@client.event
async def on_ready():
    for guild in client.guilds:
        if guild.name == GUILD:
            break

    print(
        f'{client.user} is connected to the following guild:\n'
        f'{guild.name}(id: {guild.id})'
    )

channel_intents = {}

@client.event
async def on_message(message):
    if not 'bot' in str(message.channel):
        return
    if message.author.bot:
        return
    try:
        guild_id = message.guild.id
    except:
        guild_id = 'dm'
    channel = message.channel
    channel_id = message.channel.id

    intent_id = str(channel_id) + '_' + str(guild_id)
    global channel_intents
    if intent_id not in channel_intents:
        channel_intents[intent_id] = init_intent()
        channel_intents[intent_id]['key'] = str(random.random())

    print(f'{message.author}: {message.content}')
    if len(message.content) > 0:
        if message.content[0] == '!':
            if len(message.content) > 1:
                cmd = message.content[1:]
                if cmd == 'newchat':
                    embedVar = discord.Embed(title="New chat started!", color=0x74a89b)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                    channel_intents[intent_id]['chat_gpt'] = True
                    channel_intents[intent_id]['chat_msgs'] = []
                elif cmd == 'stopchat':
                    embedVar = discord.Embed(title="Chat stopped!", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                    channel_intents[intent_id]['chat_gpt'] = False
                    channel_intents[intent_id]['chat_msgs'] = []

                elif cmd == 'currenttoken':
                    embedVar = discord.Embed(title=f"Tokens used : {channel_intents[intent_id]['current_token']}", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'currentlimit':
                    embedVar = discord.Embed(title=f"Current token limit : {channel_intents[intent_id]['token_limit']}", color=0xf75948)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'ping':
                    await channel.send('Pong! {0}ms'.format(round(client.latency*1000, 1)), reference=message, mention_author=False)

                elif cmd[:16] == 'set systemprompt':
                    prompt = cmd[16:]
                    if prompt == ' default':
                        channel_intents[intent_id]['command_txt'] = {'role' : 'system', 'content' : SECURITY_PROMPT}
                    else: channel_intents[intent_id]['command_txt'] = {'role' : 'system', 'content' : SECURITY_PROMPT + ' ' + prompt}

                    embedVar = discord.Embed(title=f"System prompt set as : {prompt[:220]}", color=0x22f5dc)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)

                elif cmd == 'currentsystemprompt':
                    prompt = channel_intents[intent_id]['command_txt']['content']

                    embedVar = discord.Embed(title=f"Current system prompt : {prompt[:220]}", color=0x22f5dc)
                    await channel.send(embed=embedVar, reference=message, mention_author=False)
                # elif cmd == 'help':
                #     embedVar = discord.Embed(title="Commands List:", color=0x22f5dc)
                #     embedVar.add_field(name="ChatGPT Commands", value="`!newchat` to start a new conversation, use the '` key' before any messages to talk to the bot.", inline=False)
                #     await channel.send(embed=embedVar)
        elif message.content[0] == '`':
            if len(message.content) > 1:
                cmd = message.content[1:]
                if channel_intents[intent_id]['chat_gpt']:
                    key_bytes = channel_intents[intent_id]['key'].encode('utf-8')

                    text_bytes = str(message.author).encode('utf-8')
                    encrypted_author = hashlib.sha256(text_bytes + key_bytes).hexdigest()
                    print(encrypted_author)

                    msg =   f'{encrypted_author}: {cmd}'
                    channel_intents[intent_id]['chat_msgs'].append({'role' : 'user', 'content' : msg})

                    if not str(message.author) in channel_intents[intent_id]['users']:
                        channel_intents[intent_id]['users'].append(str(message.author))

                    if channel_intents[intent_id]['users'] == []:
                        command_txt = [channel_intents[intent_id]['command_txt']]

                    else:
                        command_txt_list = [channel_intents[intent_id]['command_txt']['content'] + ' Instead, please refer to ']
                        for user in channel_intents[intent_id]['users']:

                            text_bytes = user.encode('utf-8')
                            encrypted_user = hashlib.sha256(text_bytes + key_bytes).hexdigest()

                            command_txt_list.append(f'{encrypted_user} as {user[:-5]}, ')
                            print(encrypted_user)
                        command_txt_list[:-1].append('.')
                        command_txt = {'role' : 'system', 'content' : "".join(command_txt_list)}
                    print(msg)
                    print(command_txt)
                    response = await generate(model=channel_intents[intent_id]['model'], messages=[command_txt] + channel_intents[intent_id]['chat_msgs'])

                    response_msg = response.choices[0].message.content
                    used_tokens = response.usage.total_tokens
                    channel_intents[intent_id]['current_token'] = used_tokens

                    print(f'ChatGPT Response : {response_msg}')
                    
                    if used_tokens > channel_intents[intent_id]['token_limit']:
                        channel_intents[intent_id]['chat_msgs'] = channel_intents[intent_id]['chat_msgs'][2:]

                    channel_intents[intent_id]['chat_msgs'].append({'role' : 'assistant', 'content' : response_msg})

                    msg_list = response_msg.split('\n')
                    num_char = 0
                    chunk_list =[]
                    for message_chunk in msg_list:
                        while(len(message_chunk) > 1990):
                            await channel.send(message_chunk[:1990], reference=message, mention_author=False)
                            message_chunk = message_chunk[1990:]

                        chunk_list.append(message_chunk)
                        num_char  = num_char + len(message_chunk)
                        if num_char > 1990 - len(chunk_list):
                            chunk_list = chunk_list[:-1]
                            num_char  = num_char - len(message_chunk)
                            await channel.send("\n".join(chunk_list), reference=message, mention_author=False)

                            num_char  = len(message_chunk)
                            chunk_list = [message_chunk]
                    await channel.send("\n".join(chunk_list), reference=message, mention_author=False)

                    # if len(response_msg) > 1900:
                    #     msg_list = response_msg.split('/n')
                    #     for message in msg_list:
                    #         await channel.send(message)
                    # else:
                    #     await channel.send(response_msg)


client.run(TOKEN)