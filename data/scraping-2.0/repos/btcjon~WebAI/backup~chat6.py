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

import logging  # import logging module

logging.basicConfig(level=logging.INFO)  # configure logging

executor = ThreadPoolExecutor(max_workers=5)


async def main():
    parser = argparse.ArgumentParser(description='Interact with Bard, Bing, and OpenAI APIs.')
    parser.add_argument('--bard', help='Send a request to the Bard API.', action='store_true')
    parser.add_argument('--bing', help='Send a request to the Bing API.', action='store_true')
    parser.add_argument('--openai', help='Send a request to the OpenAI API with gpt-3.5-turbo.', action='store_true')
    parser.add_argument('--openai16k', help='Send a request to the OpenAI API with gpt-3.5-turbo-16k.', action='store_true')
    parser.add_argument('--gpt4', help='Send a request to the OpenAI API with gpt-4.', action='store_true')
    args = parser.parse_args()

    # Here you should import your LLMs
    from LLMs.bard import bard_llm
    from LLMs.bing import bing_llm
    from LLMs.openai import openai_llm_turbo, openai_llm_turbo_16k, openai_llm_4

    langchain = LangChain([bard_llm, openai_llm_turbo, openai_llm_turbo_16k, openai_llm_4])

    # After initializing the LangChain
    print(f"LLM names after LangChain initialization: {langchain.llms.keys()}")


    # Initialize the tools
    file_tool = FileSystemTool("File Tool")

    # Initialize the ToolChain
    toolchain = ToolChain([file_tool])

    # Test writing to a file
    toolchain.use_tool("File Tool", "write_file", "test.txt", "Hello, World!")

    # Test reading from a file
    content = toolchain.use_tool("File Tool", "read_file", "test.txt")
    print(content)

    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        if args.bard:
            await langchain.process(prompt, 'Bard')
        if args.bing:
            await langchain.process(prompt, 'Bing')
        if args.openai:
            await langchain.process(prompt, 'gpt-3.5-turbo')
        if args.openai16k:
            await langchain.process(prompt, 'gpt-3.5-turbo-16k')
        if args.gpt4:
            await langchain.process(prompt, 'gpt-4')

if __name__ == "__main__":
    asyncio.run(main())
