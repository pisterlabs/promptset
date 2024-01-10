'''
chat.py
~~~~~~~
Asynchronous Chat Completion functionality.
'''

# Import libraries
import openai
import discord
from typing import Union, List
from src.core import constants
from src.memory import memory

'''
Construction Functions
'''

async def constructor_media(client: openai.AsyncOpenAI, message: discord.Message) -> list:
    """
    Constructs the uplink list.
    """
    # Construct system prompt
    uplink = [
            {
                'role'    : 'system',
                'content' : constants.CHAT_PROMPT,
            },
    ]

    # Fetch conversation context from memory queue
    context = await memory.unravel(message=message)
    # Augment uplink with context
    uplink.extend(context)

    return uplink

async def media_info(client: openai.AsyncOpenAI, message: discord.Message, media: list) -> list:
    """
    Asks GPT-4 Vision for descriptions of images provided and turns it into an extensible list of dicts.
    """
    # List to hold descriptions
    descriptions = []

    for attachment in media:
        # Components of the request
        image = {
                'url'    : attachment.url,
                'detail' : constants.VISION_DETAIL,
                }

        body = [
            {
                'type' : 'text',
                'text' : constants.VISION_PROMPT
            },
            {
                'type'      : 'image_url',
                'image_url' : image,
            },
        ]

        request = [
            {
                'role'    : 'user',
                'content' : body,
            }
        ]
        
        # Actual API call
        info = await client.chat.completions.create(
            messages   = request,
            model      = constants.VISION_MODEL,
            max_tokens = constants.VISION_MAX,
        )
        
        description = info.choices[0].message.content

        # Add description to list of descriptions
        descriptions.append(description)

        # !!! Containerize <--- Calls another function!!
        container = await containerize(message=message, descriptions=descriptions)
        
        return container


async def containerize(message: discord.Message, descriptions: list) -> list:
    """
    Converts a list of descriptions into a list of dictionaries
    """
    container = []

    for description in descriptions:
        context = {
            'role'    : 'system',
            'content' : f'{message.author.display_name} uploaded an image. Here is a description of the image:',
        }
        text = {
            'role'    : 'system',
            'content' : description,
        }

        # Append to container
        container.append(context)
        container.append(text)
    
    return container

'''
Primary API Call
'''

async def chat_completion_media(client: openai.AsyncOpenAI, message: discord.Message, media: list) -> Union[str, List]:
    """
    Sends and receives a response from the OpenAI API.
    """
    # Construct initial uplink
    uplink = await constructor_media(client=client, message=message)

    # Add media context <--- lots happening under the hood!!
    media_context = await media_info(client=client, message=message, media=media)
    # Augment uplink with media context
    uplink.extend(media_context)

    response = await client.chat.completions.create(
        messages    = uplink,
        model       = constants.CHAT_MODEL,
        temperature = constants.CHAT_TEMP,
        max_tokens  = constants.CHAT_MAX,
    )

    return response.choices[0].message.content, media_context
