import sqlite3
from datetime import datetime, timedelta
import random
import asyncio
from utils.get_conversation import get_conversation
from utils.handle_send_to_discord import generate_response
from openai import OpenAI
import openai
import concurrent.futures
from utils.exponential_backoff import exponential_backoff, get_latest_conversation
from utils.store_conversation import store_conversation
from dotenv import load_dotenv
import os
import discord

load_dotenv()
discord_token = os.getenv("DISCORD_TOKEN")
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

client = OpenAI(base_url=api_base, api_key=api_key, max_retries=0)

# New database file
db_file = 'conversation_times.db'

def create_table():
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_times (
            conversation_id TEXT PRIMARY KEY,
            last_message_time DATETIME,
            next_message_time DATETIME
        )
    """)
    conn.commit()
    conn.close()

def update_last_message_time(conversation_id):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    current_time = datetime.utcnow()
    next_message_delta = timedelta(days=random.uniform(2, 4))  # Random time between 1 to 2 days
    next_message_time = current_time + next_message_delta

    cursor.execute("""
        INSERT INTO conversation_times (conversation_id, last_message_time, next_message_time)
        VALUES (?, ?, ?)
        ON CONFLICT(conversation_id)
        DO UPDATE SET last_message_time = excluded.last_message_time,
                      next_message_time = excluded.next_message_time
    """, (conversation_id, current_time, next_message_time))
    print("Updated last message time for conversation_id", conversation_id, "to", current_time, "and next message time to", next_message_time)
    
    conn.commit()
    conn.close()

async def check_channels_for_message(message, client):
    while True:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        current_time = datetime.utcnow()
        cursor.execute("SELECT conversation_id, next_message_time FROM conversation_times")
        rows = cursor.fetchall()
        conversation_id = message.channel.id
        for row in rows:
            conversation_id, next_message_time = row
            if next_message_time and datetime.strptime(next_message_time, '%Y-%m-%d %H:%M:%S.%f') <= current_time:
                await send_reminder_message(message, client)
                update_last_message_time(conversation_id)  # Reset the next_message_time
        conn.close()
        # print("Sleeping for 2 seconds, checking again in 2 seconds")
        await asyncio.sleep(2)  # Wait for 10 minutes before checking again

async def send_reminder_message(message, client):
    is_dm = False
    if isinstance(message.channel, discord.DMChannel):
        is_dm = True
    
    conversation_id = message.channel.id
    # print(conversation_id)
    conversation = get_latest_conversation(conversation_id)
    # print(conversation)
    if is_dm:
        conversation.append(
            {
                "role": "system",
                "content": "The user hasn't sent you a message in a while. Send them a nice message to get them back into the conversation, or just continue the conversation yourself and await their return. Take into account the time of their last message, as the current time is now " + str(datetime.utcnow()) + "UTC.",
            }
        )
    else:
        conversation.append(
            {
                "role": "function",
                "name": "inactive_channel",
                "content": "Generate an engaging message to revive an inactive Discord server conversation and encourage user participation. You can mention things previously in the conversation, bring up previous users or topics, or mention things about yourself. Keep time of the last message in mind. The current time of this message is " + str(datetime.utcnow()) + "UTC.",
            }
        )
    
    # generate a response using the openai API, then send this reponse to the corresponding Discord channel.
    try:
        response = await single_generate_response(conversation)
    except openai.APIError as e:
        print(f"API call failed with error when trying to send reminder message")
        return
    response_message = response['choices'][0]['message']['content']

    # print(response_message)
    
    conversation.append(
        {
            "role": "assistant",
            "content": response_message,
        }
    )
    store_conversation(conversation_id, conversation)
    target_channel = client.get_channel(conversation_id)
    # send the response to the discord conversation with the given conversation_id
    await message.channel.send(response_message)
    # print(response)
    pass

def synchronous_generate_response(model, latest_conversation, client):
    return client.chat.completions.create(
        model=model,
        messages=latest_conversation,
        stream=False,
        allow_fallback=True,
        # premium=True
    )

async def single_generate_response(conversation):
    loop = asyncio.get_running_loop()
    
    model = "gpt-4"

    with concurrent.futures.ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(pool, synchronous_generate_response, model, conversation)

    return response

# Start the periodic task
create_table()
# run the async function send_reminer_message()
# asyncio.run(send_reminder_message("279718966543384578", message=discord.Message))

# asyncio.run(check_channels_for_message())