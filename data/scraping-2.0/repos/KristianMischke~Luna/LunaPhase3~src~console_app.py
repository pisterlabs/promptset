import asyncio
import os

from dotenv import load_dotenv

from Luna import Luna
from UsageTrackerDict import UsageTrackerDict
from chat.OpenAiChatGPT import OpenAiChatGPT
from chat.ChatMessage import ChatMessage
from LunaBrain import LunaBrain
from LunaBrainState import LunaBrainState
from plugins.TenorGif import TenorGif

load_dotenv()

usage_tracker_dict = UsageTrackerDict()

open_ai_api_key = os.getenv("OPENAI_API_KEY")
luna_brain = LunaBrain(open_ai_api_key, usage_tracker_dict, LunaBrainState())

chat_context = []

tenor_api_key = os.getenv("TENOR_API_KEY")
tenor_gif = TenorGif(tenor_api_key, "Luna", 8)


async def respond(message: str):
    print(message)
    chat_context.append(ChatMessage(role="assistant", content="/respond " + message))


async def gif(query: str):
    gif_link = tenor_gif.find_gif(query)
    print(gif_link)
    chat_context.append(ChatMessage(role="assistant", content="/respond " + query))


async def main():
    while True:
        print("")

        user_prompt = ""

        user_line = input()
        while user_line.strip() != "":
            user_prompt += user_line
            user_line = input()

        chat_context.append(ChatMessage(role="user", content=user_prompt))

        callbacks = {
            "respond": respond,
            "gif": gif
        }
        luna = Luna(chat_context, callbacks, luna_brain)
        await luna.generate_and_execute_response_commands()


if __name__ == '__main__':
    asyncio.run(main())
