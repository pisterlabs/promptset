'''
cognition
~~~~~~~~~
Cognitive functions using API requests.
'''
'''
 ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██║██╔═══██╗████╗  ██║
██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║██║   ██║██╔██╗ ██║
██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║██║   ██║██║╚██╗██║
╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ██║╚██████╔╝██║ ╚████║
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝                                                                                                                                  
'''

# Import libraries
import discord
from openai import AsyncOpenAI
from typing import Union, List
# Import modules
from src.core import constants
from src.cognition import chat, chat_media

# Initialize client with openai key
client = AsyncOpenAI(api_key=constants.OPENAI)

async def response(message: discord.Message) -> str:
    """
    Front-end for a chat completion.
    """
    response = await chat.chat_completion(client=client, message=message)
    return response

async def response_media(message: discord.Message, media: list) -> Union[str, List]:
    """
    Front-end for a chat completion with media content.
    """
    response, media_context = await chat_media.chat_completion_media(client=client, message=message, media=media)  

    return response, media_context

# Clean up multiple function passthroughts of client, message etc., in chat.py and chat_media.py