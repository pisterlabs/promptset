from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import openai
import argparse
import json
import asyncio
from BardAPI.bardapi.core import Bard
from EdgeGPT.src.EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from OpenAI_API.core import interact_with_openai
from concurrent.futures import ThreadPoolExecutor
from tools.file_system_tool import FileSystemTool
from llm import LLM
from lang_chain import LangChain
from tool_chain import ToolChain
from command_parser import CommandParser

import logging  # import logging module

logging.basicConfig(level=logging.INFO)  # configure logging

executor = ThreadPoolExecutor(max_workers=5)

# Path to the configuration file
CONFIG_FILE = "config.txt"

def get_default_llm():
    try:
        with open(CONFIG_FILE, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        # If the file does not exist, default to Bard
        return 'Bard'

def set_default_llm(llm):
    with open(CONFIG_FILE, 'w') as file:
        file.write(llm)

async def main():
    parser = argparse.ArgumentParser(description='Interact with Bard, Bing, and OpenAI APIs.')
    parser.add_argument('--bard', help='Send a request to the Bard API.', action='store_true')
    parser.add_argument('--bing', help='Send a request to the Bing API.', action='store_true')
    parser.add_argument('--openai', help='Send a request to the OpenAI API with gpt-3.5-turbo.', action='store_true')
    parser.add_argument('--openai16k', help='Send a request to the OpenAI API with gpt-3.5-turbo-16k.', action='store_true')
    parser.add_argument('--gpt4', help='Send a request to the OpenAI API with gpt-4.', action='store_true')
    parser.add_argument('--set-default', help='Set the default model', type=str)
    args = parser.parse_args()

    # If the --set-default flag is used, update the default LLM
    if args.set_default:
        set_default_llm(args.set_default)

    # Get the default LLM
    default_llm = get_default_llm()

    # Here you should import your LLMs
    from LLMs.bard import bard_llm
    from LLMs.bing import bing_llm
    from LLMs.openai import openai_llm_turbo, openai_llm_turbo_16k, openai_llm_4

    langchain = LangChain([bard_llm, bing_llm, openai_llm_turbo, openai_llm_turbo_16k, openai_llm_4])

    file_tool = FileSystemTool("File Tool")
    toolchain = ToolChain([file_tool])

    # Initialize the CommandParser
    command_parser = CommandParser(toolchain)

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break

        # Check if the prompt is a command
        if prompt.lower().startswith("!"):
            if prompt.lower() == "!help":
                CommandParser.list_commands(toolchain)
            else:
                command_parser.parse(prompt[1:])
        else:
            if args.bard or (default_llm == 'Bard'):
                await langchain.process(prompt, 'Bard')
            if args.bing or (default_llm == 'Bing'):
                await langchain.process(prompt, 'Bing')
            if args.openai or (default_llm == 'gpt-3.5-turbo'):
                await langchain.process(prompt, 'gpt-3.5-turbo')
            if args.openai16k or (default_llm == 'gpt-3.5-turbo-16k'):
                await langchain.process(prompt, 'gpt-3.5-turbo-16k')
            if args.gpt4 or (default_llm == 'gpt-4'):
                await langchain.process(prompt, 'gpt-4')

if __name__ == "__main__":
    asyncio.run(main())
