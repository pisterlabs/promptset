# bot.py
import os
import sys

import discord
import argparse
import requests
import openai

parser = argparse.ArgumentParser()
parser.add_argument("issue_number")
args = parser.parse_args()
num = int(args.issue_number)

response = requests.get(f"https://api.github.com/repos/mcgill-robotics/AUV-2024/issues/{num}")
json = response.json()
title = json["title"]
names = [assignee['login'] for assignee in json["assignees"]]

if len(names) == 0:
    sys.exit(0)

if len(names) == 1:
    names = names[0]

# if len(names) >= 1:
#     names_string = names[0]
#     for name in names[1:-1]:
#         names_string += ", " + name
#     if len(names) > 1:
#         names_string += " and " + names[-1]
# else:
#     names_string = "no assignee names provided."


# message = "Hey Clarke here. Big thanks to " + names_string \
#     + " for closing issue " + str(json["number"]) + " - " + title + ". I really appreciate your hard work!"


DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
openai.api_key = os.getenv("CHAT_GPT_TOKEN")

def chat_with_chatgpt(messages, model="gpt-3.5-turbo"):
    response = openai.chat.completions.create(
            model=model, messages=messages
        )

    message = response.choices[0].message.content
    return message

prompt = [ {"role": "system", "content":f"You are a discord bot for autonomous underwater vehicle(AUV) projet for McGill robotics, an undergraduate student\
    design team. As a discord bot you pretend to be the robot, \"Clarke\". You job is to give a short kudos to a member who just closed an\
    issue on github. You are given the issue number and the title of the issue. You are also given the names of the assignees.\
    The name(s) are {names}, the issue number is {num} and the title is {title}. Output the names as they are (even if they are weird nicknames do not change them in any way). Your message should be brief, wholesome, and funny. \
    Keep the message short and sweet. Start and end the message with beep boop. Maximum three sentences."}]


message = chat_with_chatgpt(prompt)

intents = discord.Intents.default()
client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f'{client.user} has connected to Discord!')
    await send_message()


async def send_message():
    for guild in client.guilds:
        if guild.name == "McGill Robotics":
            for channel in guild.channels:
                if channel.name == "auv-general":
                    await channel.send(message)
                    sys.exit(0)



client.run(DISCORD_TOKEN)