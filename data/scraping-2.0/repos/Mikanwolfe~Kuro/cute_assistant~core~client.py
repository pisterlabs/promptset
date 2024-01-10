import discord
from discord.ext import commands
from discord.commands import option
from datetime import datetime
import cute_assistant.core.nosql_module as db
import cute_assistant.core.database_utils as vdb
import cute_assistant.core.tokens as tk
import openai
import pprint
import secrets
import random
import re
import json
from cute_assistant.core.log import cute_logger as logger
from cute_assistant.utils.utils import remove_links, remove_phrases_from_string, format_message, format_discord_tag, format_memory

import asyncio

with open("datastore/settings.json", "r") as f:
    settings = json.load(f)

with open("datastore/config.json", "r") as f:
    config = json.load(f)

with open("datastore/responses.json", "r") as f:
    responses = json.load(f)

openai.api_key = settings["openai_api_key"]
vdb.db_bearer_token = settings["db_int_bearer_token"]

# Set up bot
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = commands.Bot(command_prefix="!", intents=intents)
client.load_extension("cute_assistant.extensions.config_cog")
status_ch = None

remove_phrases = config['removed_phrases']

@client.event
async def on_ready():
    global status_ch
    logger("discord").info(f"Logged in as {client.user} (ID: {client.user.id})")
    on_ready_msg = f"Logged in as {client.user} (ID: {client.user.id})"
    print(on_ready_msg)
    print("------")

    # Log status to dedicated announcements channel
    status_ch = client.get_channel(1102061260671045652)
    await status_ch.send(on_ready_msg)

async def handle_upload(message) -> str:
    ctx = await client.get_context(message)
    user = message.author
    async with ctx.typing():
        if message.attachments:
            for attachment in message.attachments:
                if attachment.filename.endswith('.txt') or attachment.filename.endswith('.md'):
                    content = await attachment.read()
                    content_str = remove_links(content.decode('utf-8'))
                    db.save_file_content(attachment.filename, content_str)
                    response_code = vdb.upsert(secrets.token_hex(64 // 2), str({"content" : format_discord_tag(user) + " : " + content_str, "time" : str(datetime.now().isoformat())}))
            msg_response = " > " + random.choice(responses['memory_plural'])
        if str(message.content) != "":
            response_code = vdb.upsert(secrets.token_hex(64 // 2), str({"content" : format_discord_tag(user) + " : " + str(message.content), "time" : str(datetime.now().isoformat())}))
            msg_response = " > " + random.choice(responses['memory_single'])
        
        if response_code == 200: msg_response += f"\n {random.choice(responses['memory_success'])}" 
        else: msg_response += f"\n {random.choice(responses['memory_failure'])}" 

    return msg_response


@client.event
async def on_message(message):
    ctx = await client.get_context(message)
    user = message.author

    if message.author.bot:
        return
    
    if not db.is_channel_allowed(message.channel.id):
        return
    
    if db.get_channel_type(message.channel.id) == "memory":
        msg_response = await handle_upload(message)
        await message.reply(msg_response)
        return
    if db.get_channel_type(message.channel.id) == "test":
        # Test area
        return
    
    conversation_id = db.get_most_recent_conversation(message.channel.id)['conversation_id']
    
    system_prompt = config["system_prompt"]
    gateway_prompts = config["gateway_prompts"]
    pre_prompt = f"{str(datetime.now().isoformat())} : {config['pre_prompt']}"
    new_message = format_message(str(message.content), "user",  client.get_user(int(user.id)), preprompt=pre_prompt)
    
    # Get long term memory from vdb
    chunks_response = vdb.query_database(message.content)
    selected_memories, distant_memories = tk.get_memory_until_token_limit(chunks_response, 1024)

    for memory in selected_memories:
        db.delete_memory(memory['id'])
        db.add_memory(memory['id'], memory['text'])

    print(" --- SELECTED MEMORY --- ")
    pprint.pprint(selected_memories)
    print(" --- END --- ")
    memory_shards = []

    if config['memory_log']:
        for ch in config['memory_log']:
            send_ch = client.get_channel(ch)
            for memory in selected_memories:
                await send_ch.send('> ' + memory['text'])
                #react to delete these heheh

    for result in selected_memories:
        memory_shards.append(result["text"])

    for result in selected_memories:
        memory_shards.append(result["text"])

    distant_shards = []
    for result in distant_memories:
        distant_shards.append(result["text"])
            
    mem_messages = [format_memory("user", _msg) for _msg in memory_shards]
    dist_messages = [format_memory("user", _msg) for _msg in distant_shards]
    memory_tokens = tk.get_num_tokens(mem_messages)
    distant_tokens = tk.get_num_tokens(dist_messages)
    print(f"Memory Tokens: {memory_tokens}, Distant Tokens: {distant_tokens}")

    print(" --- MEMORY --- ")
    pprint.pprint(mem_messages)
    print(" --- END --- ")

    system_prompt_tokens = tk.get_num_tokens([system_prompt])
    gateway_prompts_tokens = tk.get_num_tokens(gateway_prompts)
    memory_tokens = tk.get_num_tokens(mem_messages)
    query_tokens = tk.get_num_tokens([new_message])
    
    max_tokens = 4096 - 1200 - system_prompt_tokens - gateway_prompts_tokens - memory_tokens - query_tokens
    last_messages = tk.get_messages_until_token_limit(conversation_id, max_tokens)
    
    prev_messages = [format_message(_msg['message'], _msg['role'], client.get_user(int(_msg['user_id']))) for _msg in last_messages]
    last_messages_tokens = tk.get_num_tokens(prev_messages)

    # Message format
    if prev_messages: 
        messages = [system_prompt] + gateway_prompts + mem_messages + prev_messages + [new_message]
    else:
        messages = [system_prompt] + gateway_prompts + mem_messages  + [new_message]


    memory_prompt = {
        "role" : "user",
        "content" : f"You are to shorten {len(distant_shards)} pieces of information encapsulated in <> into {max(len(distant_shards)-1,1)} or less pieces also encapsulated in <> organised by ideas. Split and re-word if ideas are combined such that the output pieces contain related ideas."
    }
    # Memory Format
    for shard in distant_shards:
        filtered_shard = shard.replace('<', '').replace('>', '')
        memory_prompt['content'] = f"{memory_prompt['content']} <{filtered_shard}>" 


    memory_messages = [
            {
            "role" : "system",
            "content" : "You are a text shortening AI, using your understanding of natural language to ensure summaries are faithful to the original content and written in Australian English. The shortened text contains wording from the original where possible. If the piece is an encoded string that is not natural language, please remove the piece entirely."
        },
        memory_prompt
    ]

    max_mem_tokens = 4096 - tk.get_num_tokens(memory_messages)

    print(" --- MESSAGES --- ")
    pprint.pprint(messages)
    print(" --- END --- ")

    max_tokens = max_tokens - last_messages_tokens - 1

    config_temp = db.get_channel_setting(message.channel.id, "config_temp", default=config['default_temp'])
    config_freq = db.get_channel_setting(message.channel.id, "config_freq", default=config['default_freq'])
    config_pres = db.get_channel_setting(message.channel.id, "config_pres", default=config['default_pres'])

    print(f"Total tokens: 4096 - {system_prompt_tokens} SP - {gateway_prompts_tokens}GPT - {memory_tokens} MT - {query_tokens} QT - {last_messages_tokens} LMT = {max_tokens} Tokens left. Used tokens: {4096 - max_tokens}")

    async with ctx.typing():
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=max_tokens + 1200, # we took 96 off just before
            temperature=config_temp,  # High temperature leads to a more creative response.
            frequency_penalty=config_freq,  # High temperature leads to a more creative response.
            presence_penalty=config_pres,  # High temperature leads to a more creative response.
        )

        
        msg_response = response["choices"][0]["message"]["content"]
        msg_response = remove_phrases_from_string(remove_phrases, msg_response)
        db.add_message(conversation_id, str(user.id), 'user', str(message.content))
        db.add_message(conversation_id, str(client.user.id), 'assistant', str(msg_response))

        
        status_msg = f"{message.author.name}: {message.content}"

        # We should focus on user experience next and include user descriptions, pre config, etc.
        vdb.upsert(secrets.token_hex(64 // 2), str({"query" : format_discord_tag(user) + " : " + str(message.content), "response": format_discord_tag(client.user) + " : " +str(msg_response), "time" : str(datetime.now().isoformat())}))
    await message.channel.send(msg_response)

    if (len(distant_shards) >=1):

        memory_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=memory_messages,
            max_tokens=max_mem_tokens - 1, # we took 96 off just before
            temperature=config_temp,  # High temperature leads to a more creative response.
            frequency_penalty=config_freq,  # High temperature leads to a more creative response.
            presence_penalty=config_pres,  # High temperature leads to a more creative response.
        )

        mem_response = memory_response["choices"][0]["message"]["content"]


        new_distant_shards = re.findall(r'<(.*?)>', mem_response)
        new_distant_shards = [remove_links(s) for s in new_distant_shards]

        print(" --- MEM PROMPT --- ")
        pprint.pprint(memory_messages)
        print(" --- END --- ")

        print(" --- MEM RESPONSE --- ")
        pprint.pprint(new_distant_shards)
        print(" --- END --- ")

        print(" --- UPDATING DISTANT MEMORIES --- ")
        ids = []
        for memory in distant_memories:
            ids.append(str(memory['id']))
            db.delete_memory(memory['id'])
            #maybe get IDs back to update database.

        print("Deleting: " + str(ids))
        vdb.delete_vectors(ids)
        
        vdb.upsert_texts(new_distant_shards)

        print(" --- END --- ")
    else: 
        print(" --- NO DISTANT MEMORIES RETRIEVED --- ")


def run_bot(token):
    client.run(token)