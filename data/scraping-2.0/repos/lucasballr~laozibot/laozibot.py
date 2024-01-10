#! /usr/bin/env python

import discord
from discord.ext import commands
import os
import io
import requests
import openai, config
import asyncio
import sqlite3
import re
from datetime import datetime, timedelta
import json
from w6 import get_user, get_weather, get_location, update_location, add_user
from gpt import ask_gpt
import typing
import functools

openai.api_key = config.OPENAI_API_KEY
discord_api_key = config.DISCORD_API_KEY

# Connect to the database
db_conn = sqlite3.connect('laozidb.db')
db_conn.execute('CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, remind_at INTEGER, content TEXT)')
db_conn.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, location TEXT)''')
command_list = ['!r', '!w', '!clear']
init_message = [{'role':'system', 'content':'You are a helpful AI assistant named laozibot. You must respond to every message in under 2000 characters, so make sure each message contains all of the necessary information.'}]
reminders = {}
messages = init_message
users = {}

# Define  bot's intents
intents = discord.Intents.all()
intents.members = True  # Subscribe to the privileged members intent

# Create a new instance of the bot client
client = commands.Bot(command_prefix='!', intents=intents)

def extract_code(text):
    pattern = r'(```[\s\S]+?```)'  # Add capturing group with parentheses
    segments = re.split(pattern, text)
    return segments

def split_string(input_string):
    max_size = 2000
    # check if there is a code block in the output
    if '```' in input_string:
        input_string = extract_code(input_string)
    else:
        input_string = [input_string]

    # Loop through the resulting strings and split the string if it is too long.
    new_strings = []
    for s in input_string:
        if len(s) >= max_size:
            s = s.split('\n')
            mes = ""
            for line in s:
                if len(mes) + len(line) >= max_size:
                    new_strings.append(mes)
                    mes = line + "\n"
                else:
                    mes += line + "\n"
            new_strings.append(mes)
        else:
            new_strings.append(s)

    # Return the list of strings
    return new_strings

async def schedule_reminder(user, duration, content, reminder_id):
    # Wait until the reminder is due
    await asyncio.sleep(duration)

    # Send the reminder to the user
    await user.send(f'Reminder: {content}')

    # Delete the reminder from the database
    db_conn.execute('DELETE FROM reminders WHERE id = ?', (reminder_id,))
    db_conn.commit()

    # Check for any remaining reminders that are due
    for row in db_conn.execute('SELECT * FROM reminders WHERE remind_at <= ?', (int(datetime.utcnow().timestamp()),)):
        user_id, remind_at, content, remaining_reminder_id = row
        user = await client.fetch_user(int(user_id))
        duration = max(remind_at - int(datetime.utcnow().timestamp()), 0)
        asyncio.ensure_future(schedule_reminder(user, duration, content, remaining_reminder_id))

def remind(message):
    # Parse the reminder duration and message content
    duration, content = message.content[3:].split(' ', 1)
    # Convert the duration (in minutes) to seconds
    if duration.endswith('s'):
        duration = int(duration[:-1])
    elif duration.endswith('m'):
        duration = int(duration[:-1]) * 60
    elif duration.endswith('h'):
        duration = int(duration[:-1]) * 60 * 60
    elif duration.endswith('d'):
        duration = int(duration[:-1]) * 60 * 60 * 24
    # Schedule the reminder
    remind_at = int(message.created_at.timestamp()) + duration
     # Store the reminder in the database and schedule it
    cursor = db_conn.cursor()
    cursor.execute('INSERT INTO reminders (user_id, remind_at, content) VALUES (?, ?, ?)', (str(message.author.id), remind_at, content))
    db_conn.commit()
    reminder_id = cursor.lastrowid
    cursor.close()
    asyncio.ensure_future(schedule_reminder(message.author, duration, content, reminder_id))

def weather(message):
    cmd = message.content.split(' ')
    user = message.author.id

    if len(cmd) == 1:
        if not get_user(db_conn, user):
            return "You're not registered, to register type '!w <location>'"
        elif not get_location(db_conn, user):
            return "You need to add a location for yourself '!w <location>'"
    elif len(cmd) > 1:
        location = cmd[1]
        if not get_user(db_conn, user):
            add_user(db_conn, user, location)
        else:
            update_location(db_conn, user, location)
    else:
        return "Invalid command"

    return "Location updated"

def clear_chat(message):
    user = message.author.id
    c = db_conn.cursor()
    c.execute("DELETE FROM history WHERE name = ? OR rec = ?", (user, user))
    db_conn.commit()
    return "Chat cleared"

def run_command(message):
    cmd = message.content.split(' ')[0]
    if cmd == '!r':
        remind(message)
        return "Reminder Set"
    elif cmd == '!w':
        return weather(message)
    elif cmd == '!clear':
        return clear_chat(message)
    else:
        return "Command Error. Idk how you got here."

# This is essential for any blocking function being run during message response.
async def run_blocking(blocking_func: typing.Callable, *args, **kwargs) -> typing.Any:
    # Runs a blocking function in a non-blocking way
    func = functools.partial(blocking_func, *args, **kwargs) # `run_in_executor` doesn't support kwargs, `functools.partial` does
    return await client.loop.run_in_executor(None, func)

@client.event
async def on_ready():
    print("Logged in as {0.user}".format(client))
    # Remember the reminders
    for row in db_conn.execute('SELECT * FROM reminders WHERE remind_at > ?', (int(discord.utils.time_snowflake(datetime.utcnow() - timedelta(minutes=1))),)):
        user_id, remind_at, content = row[1:]
        user = await client.fetch_user(int(user_id))
        duration = max(remind_at - int(datetime.utcnow().timestamp()), 0)
        asyncio.ensure_future(schedule_reminder(user, duration, content, row[0]))

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Check if message is in a DM or the message starts with !g
    if isinstance(message.channel, discord.DMChannel) or message.content.startswith('!g'):
        print("{}: {}".format(message.author.name, message.content))
        if message.content.split(' ')[0] in command_list:
            text = run_command(message)
            print("laozibot: {}".format(text))
            embed = discord.Embed()
            embed.add_field(name="Command Response", value=text, inline=False)
            await message.channel.send(embed=embed)
        else:
            loc = get_location(db_conn, message.author.id)
            info = "Current Location: {}, {}".format(loc, get_weather(loc))
            async with message.channel.typing():
                gpt_res = await run_blocking(ask_gpt, message, info)
            output = split_string(gpt_res)
            for i in output:
                if i != "":
                    print("laozibot: {}".format(i))
                    await message.channel.send(i)

    # This checks if a command has been run, but I can replace this with commands.Bot
    elif message.content.split(' ')[0] in command_list:
        print("{}: {}".format(message.author.name, message.content))
        run_command(message)

client.run(discord_api_key)
