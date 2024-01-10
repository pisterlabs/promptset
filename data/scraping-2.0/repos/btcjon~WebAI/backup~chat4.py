from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import argparse
import json
import asyncio
from BardAPI.bardapi.core import Bard
from EdgeGPT.src.EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from OpenAI_API.core import interact_with_openai
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

def clean_output(output):
    # Your clean_output function here

async def chat_with_bard():
    loop = asyncio.get_event_loop()
    bard = Bard()
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        response = await loop.run_in_executor(executor, bard.get_answer, prompt)
        cleaned_response = clean_output(response)
        print(f"Bard: {cleaned_response}")

async def chat_with_bing():
    with open('cookies.json', 'r') as f:
        cookies = json.load(f)
    bing = await Chatbot.create(cookies=cookies)
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            await bing.close()
            break
        response = await bing.ask(prompt, conversation_style=ConversationStyle.creative)
        cleaned_response = clean_output(response)
        print(f"Bing: {cleaned_response}")

async def chat_with_openai():
    model = "gpt-3.5-turbo"  # The default model
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        response = await interact_with_openai(prompt, model)
        cleaned_response = clean_output(response)
        print(f"OpenAI: {cleaned_response}")


async def main():
    parser = argparse.ArgumentParser(description='Interact with Bard, Bing, ChatGPT, and OpenAI APIs.')
    parser.add_argument('--bard', help='Send a request to the Bard API.', action='store_true')
    parser.add_argument('--bing', help='Send a request to the Bing API.', action='store_true')
    parser.add_argument('--chatgpt', help='Send a request to the ChatGPT API.', action='store_true')
    parser.add_argument('--openai', help='Send a request to the OpenAI API.', action='store_true')
    args = parser.parse_args()

    if args.bard:
        await chat_with_bard()  # New code for Bard interaction

    if args.bing:
        await chat_with_bing()  # New code for Bing interaction

    if args.chatgpt:
        pass  # This part can be filled later as per user's requirement

    if args.openai:
        await chat_with_openai()  # New code for OpenAI interaction

if __name__ == "__main__":
    asyncio.run(main())
