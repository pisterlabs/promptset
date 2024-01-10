"""
Description: Runs Buck Shlegeris's and Fabien Roger's Bash Scaffold.
Purpose: For Constellation's Astra Fellowship Winter 2022-2023.
Features: streaming API, model response comes in token-by-token.
By: Naomi Bashkansky

Requires a config.ini file with the following format:
[openai]
api_key = sk-c...a
model = gpt-4

TO RUN, TYPE THE FOLLOWING INTO THE TERMINAL:
chainlit run bash_scaffold.py -w
^ If you want "cd" to work, run with the absolute path instead.
"""

# Imports
import chainlit as cl
import openai
import os
import configparser
import subprocess
import sys

# Set up OpenAI API key and model
config = configparser.ConfigParser()
config.read('config.ini')
openai.api_key = config['openai']['api_key']
model_name = config['openai'].get('model', fallback='gpt-4') 

# Current working directory
cwd = os.getcwd()

# System prompt (for Windows terminal)
system_prompt = f"""
You are being run in a scaffold in a Windows terminal. When you want to run a command, write it in a <shell> XML tag. You will be shown the result of the command and be able to run more commands. Other things you say will be sent to the user. In cases where you know how to do something, don't explain how to do it, just start doing it by emmitting bash commands one at a time. Remember that you can't interact with stdin directly, so if you want to e.g. do things over ssh you need to run commands that will finish and returning control to you rather than blocking on stdin. Don't wait for the user to say okay before suggesting a command to run. If possible, don't include the explanation, just say the command. 

To go to home directory, use the name of the starting working directory ({cwd}) to go to C:\\Users\\<YourUserName>.

If you can't do something without assistance, please suggest a way of doing it without assistance anyway. 

Remember that you're using Windows terminal, and that you should ONLY give responses in the form <shell>command</shell>. Do not say anything else.
"""

# Initializing the messages fed into the AI. 
# Switch between user asking for question and user providing permission.
history = [{"role": "system", "content": system_prompt}]
awaiting_permission = False
recent_command = ""

# When the chat starts, print out the system prompt.
@cl.on_chat_start
async def init():
    init_str = f"<START SYSTEM MESSAGE>\n{system_prompt}\n<END SYSTEM MESSAGE>"
    init_message = cl.Message(content=init_str)
    await init_message.send()

# Helper function to run shell commands and return the output.
def run_shell_command(command):
    # Handle cd commands separately, since they don't work with subprocess.
    command_split = command.split()
    if command_split[0] == 'cd':
        try:
            if len(command_split) == 1:
                return os.getcwd()
            else:
                os.chdir(command_split[1])
                return os.getcwd()
        except OSError:
            return str(sys.exc_info()[1])
    else:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout if result.stdout else result.stderr

# When the user sends a request, send it to the AI.
# The AI generates a bash command, which is then run after user gives permission.
@cl.on_message
async def main(message: str):
    global awaiting_permission
    global recent_command

    # The model sees the entire history: 
    # system, user, and assistant messages, and command outputs.
    if not awaiting_permission:
        history.append({"role": "user", "content": message.content})  

        # Generate model response.
        response = openai.ChatCompletion.create(
            model = model_name,
            messages = history,
            temperature = 1,
            stream = True,
        )

        # Send the model response to the user, one token at a time.
        msg = cl.Message(content="")
        for token in response:
            delta = token['choices'][0]['delta']
            if "content" in delta:
                await msg.stream_token(token['choices'][0]['delta']['content'])        
        await msg.send()

        history.append({"role": "assistant", "content": msg.content})

        # If the model response is a shell command, ask for permission to run it.
        if msg.content[:7] == "<shell>" and msg.content[-8:] == "</shell>":
            awaiting_permission = True
            recent_command = msg.content[7:-8]
            enter_msg = cl.Message(content="Run command? Y/N")
            await enter_msg.send()
    else:
        # If the user says yes, run the command and send the output to the user.
        awaiting_permission = False
        yeses = ["Y", "y", "yes", "Yes", "YES"]
        if message.content in yeses:
            output = run_shell_command(recent_command)
            await cl.Message(content=output).send()
            history.append({"role": "assistant", "content": output})
