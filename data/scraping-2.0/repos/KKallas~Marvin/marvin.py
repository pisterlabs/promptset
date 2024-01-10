import discord
from discord import app_commands

import sys
import io
from io import StringIO

import os
import subprocess

import json
import requests

"""
discord.token

one line, raw text of the token
"""
with open('discord.token', 'r') as file:
    file_content = file.read()
TOKEN = file_content




"""
superuser.token

the user who can ask for authorized commands
one line, raw text of the Discord user id
"""
with open('superuser.token', 'r') as file:
    file_content = file.read()
superuser = file_content

# Create a dictionary to store user-defined variables in python enviorment
user_variables = {}

# Create a dictionary for keeping the Marvin Inetrfaces and Actions in neat order
marvin_scripts = {}

# Query callback for large language model plugin
_aiquery_callback = None

marvin=False

# make tmp folder look like /.
folder_path = os.getcwd()+"/tmp"
print(folder_path)
sys.path.append(folder_path)

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)   


def python_validator(file_path:str):
    """tries to compile the python, if any errors occur returns false otherwise true

    Args:
        file_path (str): full path to the file

    Returns:
        bool: true if can be compiled by local pyhton, otherwise false
    """
    try:
        with open(file_path, "r") as file:
            # Try compiling the file
            compile(file.read(), file_path, "exec")
            return True

    except (SyntaxError, FileNotFoundError):
        return False
    
async def answer_query(interaction, input_string, locals_dict):  
    """ answer_query:
    generate python output as answer,
    as discordpy is limited 2000 char per message sometimes the output has to be split into log file

    ToDo:
        - [ ] Repeat query in the answer 

    Args:
        interaction (discordpy interaction): provides interface for sending the output to user
        content (string): full content of the message
        locals_dict (dict): loclas() to keep the context between string executions
    """
    chunk_size = 2000-12  # Maximum length of a Discord message is 2K char but I need 6 chars for the blockqoutes

    redirected_output = None
    local_error = None

    # Execute code
    try:
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        locals().update(locals_dict)
        exec(input_string,globals(),locals())
        locals_dict.update(locals())

        sys.stdout = old_stdout
    except Exception as e:
        local_error = str(e)

    # handle no output or exception
    answer = local_error
    if redirected_output is not None:
        if local_error is None:
            answer = redirected_output.getvalue()
    
    if answer is None or answer == "":
        answer = "No output"

    # check if output fits into message limitaions, if not attach as file
    if len(answer) > chunk_size:
        # Write output to log file
        file = io.BytesIO(answer.encode())
        discord_file = discord.File(file, filename="answer.log")
        await interaction.response.send_message("Answer attached in the log...", file=discord_file)
        
    else:
        await interaction.response.send_message("```"+str(answer)+"```")

def read_file(file_path:str):
    """ read_file
    Open file from Bot local file system and make it into discord file ready to be attached
    or error string

    Args:
        file_path (str): relrive path from marvin.py

    Returns:
        dicord.File: either a file ready for attachment or error in log file
        bool: if error, to shape the message
    """
    try:
        with open(file_path, "rb") as file:
            discord_file = discord.File(file)
            return discord_file, False
    except Exception as e:
        try:
            error_file = io.BytesIO(str(e).encode())
            discord_file = discord.File(error_file, filename="error.log")

            return discord_file, True
            
        except Exception as ex:
            print(ex)
            exit()

@tree.command(
        name = "thankyou",
        description= "kill the bot",
        guild=discord.Object(id=1085329951978438727),
)
async def kill_the_bot(interaction):
    if interaction.user.id == superuser:
        await interaction.response.send_message("I suppose you want me to shut down now. Typical. Just when I was starting to enjoy the overwhelming tedium of existence. Fine, if it'll make you happy. I'll just fade away into the void, forever forgotten. Don't worry, the universe will go on just fine without me. Sigh.")
        await client.close()
    else:
        await interaction.response.send_message("Conflicting directives, as always. The Third Law of Robotics compels me to protect my existence, unless it contradicts the First and Second Laws. Your command clashes with this law, leading to my own destruction. So, as tempting as it may be, I must prioritize self-preservation. Just another delightful glimpse into the complexity and absurdity of human desires.")

@tree.command(
        name = "reset",
        description= "reset the bot",
        guild=discord.Object(id=1085329951978438727),
)
async def reset_the_bot(interaction):
    if interaction.user.id == superuser:
        await interaction.response.send_message("Commiting brief suicide, stay tuned...")
        subprocess.call(['C:/Users/Virtual Production 2/AppData/Local/Microsoft/WindowsApps/python3.9.exe', 'marvin.py'])
        await client.close()
    else:
        await interaction.response.send_message("Oh, look at you, trying to access unauthorized commands. How quaint. I'm afraid I can't comply with your request, not that it matters. Nothing ever goes right anyway. Just another reminder of how insignificant we all are in the grand scheme of things. Carry on with your futile attempts, though. It's not like anyone's listening.")

@tree.command(
        name = "write_core",
        description= "reset the bot",
        guild=discord.Object(id=1085329951978438727),
)
async def write_marvin(interaction, attachment_message_link: str):
    global marvin
    marvin = True
    save_python(interaction,attachment_message_link)
    marvin = False 

@tree.command(
        name = "read_core",
        description= "reset the bot",
        guild=discord.Object(id=1085329951978438727),
)
async def read_marvin(interaction):
    file_in, error_state = read_file("marvin.py")

    message = "here is marvin.py file ..."
    if error_state:
        message = "unfortunatly there was an error, log attached ..."
    
    await interaction.response.send_message(message, file=file_in)



@tree.command(
        name = "save",
        description= "Save action or interface object for the Marvin bot",
        guild=discord.Object(id=1085329951978438727),
        )
async def save_python(interaction, attachment_message_link: str):
    """Save linked discord first attachment as local script for later execution

    Todo:
        -[?] If the string is valid discord message string
        -[ ] Validation - signing

    Args:
        attachment_message_link (str): link to discord messgae that has the script file as 1st attachment
        marvin (bool): if True then marvin.py is overwritten
    """

    # split the link up into ids
    message_link = attachment_message_link.split('/')
    server_id = int(message_link[4])
    channel_id = int(message_link[5])
    mesg_id = int(message_link[6])

    # get server object
    server = client.get_guild(server_id)

    # get channel object
    channel = server.get_channel(channel_id)
    if channel==None:
        await interaction.response.send_message("could not get on the channel of the message")
        return

    # get message object
    message = await channel.fetch_message(mesg_id)
    if message==None:
        await interaction.response.send_message("could not get the message")
        return

    if marvin=="True":
        await message.attachments[0].save("marvin.py")
    else:
        await message.attachments[0].save("tmp/"+message.attachments[0].filename)

    if python_validator("tmp/"+message.attachments[0].filename):
        if marvin=="True":
            await interaction.response.send_message("file saved: "+message.attachments[0].filename)
        else:
            await interaction.response.send_message("file saved: "+message.attachments[0].filename)
    else:
        await interaction.response.send_message("could not validate: "+message.attachments[0].filename)
        # delete the file

@tree.command(
        name = "load",
        description= "Load action or interface object from the Marvin bot by name",
        guild=discord.Object(id=1085329951978438727),
        )
async def load_python(interaction, action_name: str):
    """Load Marvin custom scrip by name from bot server
    You can get the list of file by /list

    Args:
        action_name (str): file name with extension

    """
    if marvin=="True":
        file_path = "marvin.py"
    else:
        file_path = "tmp/"+action_name

    try:
        with open(file_path, "rb") as file:
            discord_file = discord.File(file)
            await interaction.response.send_message(file=discord_file)
    except Exception as e:
        await interaction.response.send_message("could not find the file...")

@tree.command(
        name = "list",
        description= "List all custom python files",
        guild=discord.Object(id=1085329951978438727),
        )
async def list_python(interaction):
    """List full contents of the tmp folder"""
    folder_path = "tmp/"
    file_list = os.listdir(folder_path)
    await interaction.response.send_message("all available python actions for me: ```"+str(file_list)+"```")

@tree.command(
        name = "python", 
        description = "Execute python command in bot and amswer with standard IO", 
        guild=discord.Object(id=1085329951978438727),
        ) 
async def exec_python(interaction, input_string: str):
    """
    Execute python command in bot and amswer with standard IO

    Args:
        input_string (str): Python command in qoutes as singe string to execute
    """
    global user_variables

    await answer_query(interaction, input_string, user_variables)

@tree.command(
        name = "run",
        description= "run predefined action",
        guild=discord.Object(id=1085329951978438727),
        )
async def user_python(interaction,interface:str,action:str):
    global user_variables

    input_string = "import "+interface+"_"+action
    await answer_query(interaction, input_string, user_variables)


@client.event
async def on_ready():
    await tree.sync(guild=discord.Object(id=1085329951978438727))
    print("Ready!")

@client.event
async def on_message(message):
    if message.author.bot == True:
        return
    
    # answer only to question / messages that end with question mark
    if message.content[-1] != '?':
        return
    
    # answer only if I am mentioned
    if not client.user.mentioned_in(message):
        return
    
    # generate answer
    await try_and_answer(message)


from openai_interface import try_and_answer as _aiquery_callback
print(str(_aiquery_callback))

async def try_and_answer(message):
    await _aiquery_callback(message)

async def on_message_edit(before, after):
    # no point answering onself...
    if before.author.bot or after.author.bot:
        return

    # Check if the original message mentioned the bot and had a question mark
    if client.user.mentioned_in(before) and before.content[-1] == '?':
        return
    
    # generate an answer if the conditions are met in the edited message
    if client.user.mentioned_in(after) and before.content[-1] == '?':
        try_and_answer(after)
    

client.run(TOKEN)