import os
import discord
import openai
import requests
import shutil
import random
import json
from dotenv import load_dotenv

#TODO add -title option to change title after thread has been made

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')
openai.api_key = os.getenv("OPENAI_API_KEY")

# hardcoded intents necessary for the bot to work
intents = discord.Intents.default()
intents.members = True
intents.messages = True
intents.message_content = True

client = discord.Client(intents=intents)

CALL_RESPONSE_LIMIT = 4000
TOPIC = ""
MODEL = "gpt-4"
# this bot should keep this to False for now, should create a seperate bot for this
add_personality = False
main_prompt = "You’re a kind helpful assistant."
personality_prompt = ""
outputChannel = "gpt-4-threads"
# an array of all channels that the bot listens to
channels = ["gpt-4-threads"]

@client.event
async def on_ready():
    # announce that the bot is active
    print(f'{client.user} has connected to Discord!')

@client.event
async def on_message(message):
    global personality_prompt 
    global main_prompt
    global skyboxStyle
    
    #define all channels
    output = discord.utils.get(client.get_all_channels(), name=outputChannel)   

    #check if the message is from the bot
    if message.author == client.user:
        return
    
    if message.content.lower().startswith("!"):
        return
    
    # if message is from a bot , ignore
    if message.author.bot:
        return
    

    # developer notes
    if message.content.lower() == "?developerlog":
        developer_message = "Version 0.3: \nFixed Image Generation; should be better at handeling large number of requests\nAdded -title option to ?startchat command to choose the title of the thread \n"
        await message.channel.send(developer_message)
        await message.delete()
        return
    

    # define command to start a thread
    if message.content.lower().startswith("?startchat") and message.channel.name in channels:
        # check if -title is in the message
        title = ""
        if "-title" in message.content.lower():
            # parse the prompt after it and set it as the title
            title = message.content.lower().replace("?startchat -title" , "")

        thread_start = await output.send('Starting a thread')
        thread = await thread_start.create_thread(name = title if title != "" else message.author.name)
        await thread.send("Type ?help to get started\n\nHello " + message.author.name + ", what would you like to talk about?")
        # delete the message that started the thread
        await message.delete()
        return
    
    # defina a command to change the tile of the thread
    if message.content.lower().startswith("?changetitle") and (message.channel.type == discord.ChannelType.public_thread or message.channel.type == discord.ChannelType.private_thread):
        # parse the prompt after it and set it as the title
        title = message.content.lower().replace("?changetitle" , "")
        await message.channel.edit(name = title)
        await message.delete()
        return
    
    if message.content.lower() == "?deletethread":
        if message.channel.type == discord.ChannelType.public_thread or message.channel.type == discord.ChannelType.private_thread:
            await message.channel.delete()

        return
    
    # clears all messages except for pinned ones
    if message.content.lower()  == "?clearall":
        def not_pinned(m):
            return not m.pinned
        await message.channel.purge(limit=None, check=not_pinned)
        return

    # if the message was sent in a thread, get the history of the thread
    if message.channel.type == (discord.ChannelType.public_thread or message.channel.type == discord.ChannelType.private_thread) and message.channel.parent.name in channels:
        thread = message.channel
        messages = [message async for message in message.channel.history(limit=None)]
        typing_msg = await thread.send(getThinkingMsg())
        messages.reverse()
        contentBlob = ""
        for message in messages:
            contentBlob += message.content + "\n"
        try:
            response = callGPT(contentBlob)
            await typing_msg.delete()
            if len(response) >= 1999:
                chunks = [response[i:i+1999] for i in range(0, len(response), 1999)]
                for chunk in chunks:
                    await thread.send(chunk)
            else:
                await thread.send(response)
        except Exception as e:
            print(e)
            try:
                await thread.send(f"An error occurred: {e}")
            except Exception as e:
                print(e)
        return
    
   

# this function generates the response from gpt
def callGPT(input):
    tempPersonality = ""
    if add_personality:
        tempPersonality = personality_prompt

    messages = [
        {"role": "system", "content" : main_prompt + " " + tempPersonality}
    ]
    savedConversation = []
    userInput = input
    content = userInput
    messages.append({"role": "user", "content": content})

    completion = gpt_conversation(messages)
    chat_response = completion.choices[0].message.content

    if add_personality:
        chat_response = personalityGen(chat_response)
    
    messages.append({"role": "user", "content": chat_response})
    return chat_response
    
# helper function that returns a handles the conversation input with gpt
def gpt_conversation(conversation_log):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversation_log
    )
    return response


# this function adds a random thinking message to the thread
def getThinkingMsg():
    thinking_msgs = [
        "Thinking...",
        "Generating a response...",
        "Processing your request...",
        "Let me check...",
        "Let me think for a moment...",
        "Searching for the answer...",
        "Just a moment...",
        "I'm working on it...",
        "I'm on it...",
    ]
    return random.choice(thinking_msgs)

# helper function to send the prompt to the skybox API
def generateSkybox(prompt):
    jsonPayload = { "prompt": prompt }
    if getSkyboxStyleID(skyboxStyle) != -1:
        styleID = getSkyboxStyleID(skyboxStyle)
        jsonPayload = { "prompt": prompt, "skybox_style_id": styleID }
        print("style: " + skyboxStyle)
    print(json.loads(str(jsonPayload).replace("'", '"')))
    requests.post("http://localhost:5000", json = jsonPayload)
    print("sent request to server")
    return

# run the bot    
client.run(TOKEN)