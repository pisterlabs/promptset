from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

import openai
import argparse
import json
import asyncio
import pprint
from BardAPI.bardapi.core import Bard
from EdgeGPT.src.EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from OpenAI_API.core import interact_with_openai
from concurrent.futures import ThreadPoolExecutor

from captcha import solve_captcha

executor = ThreadPoolExecutor(max_workers=1)

async def chat_with_bard():
    loop = asyncio.get_event_loop()
    bard = Bard()
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        response = await loop.run_in_executor(executor, bard.get_answer, prompt)
        message = response['content']  # Extract the desired message
        print(f"Bard: {message}")  # Print the message

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
        message = response['item']['messages'][-1]['text']  # Extract the desired message
        print(message)

async def chat_with_openai():
    while True:
        prompt = input("You: ")
        if prompt.lower() in ["quit", "exit"]:
            break
        response = await interact_with_openai(prompt, "gpt-3.5-turbo")
        print(f"OpenAI: {response}")

async def interact_with_openai(prompt, model):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content']

async def main():
    parser = argparse.ArgumentParser(description='Interact with Bard, Bing, ChatGPT, and OpenAI APIs.')
    parser.add_argument('--bard', help='Send a request to the Bard API.', action='store_true')
    parser.add_argument('--bing', help='Send a request to the Bing API.', action='store_true')
    parser.add_argument('--chatgpt', help='Send a request to the ChatGPT API.', action='store_true')
    parser.add_argument('--openai', help='Send a request to the OpenAI API.', action='store_true')
    args = parser.parse_args()

    if args.bard:
        await chat_with_bard()

    if args.bing:
        await chat_with_bing()

    if args.chatgpt:
        pass  # This part can be filled later as per user's requirement

    if args.openai:
        await chat_with_openai()

if __name__ == "__main__":
    asyncio.run(main())
