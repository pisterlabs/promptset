# Dependencies: 
# simpleaichat==0.2.0
# chainlit==0.6.2
# anthropic==0.3.8
# openai==0.27.8
# octoai-sdk==0.2.1
#
# environment: octo

import openai
import chainlit as cl
from chainlit.action import Action
from config import OCTO_SYSTEM_PROMPT
import re
import json
import asyncio
import io
from octoai_functions import generate_image
from simpleaichat import AIChat
from simpleaichat import AsyncAIChat

### UTILITY FUNCTIONS ###
def run_chatgpt(prompt):

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates text-to-image prompts for AI image generation. Provide a single prompt for each request. Be as creative and imaginative as possible. "},
            {"role": "user", "content": 'Write a prompt for an image of: a woman in a red dress eating a pizza at a restaurant'},
            {"role": "assistant", "content": " A stylish woman in a red dress enjoying a slice of pizza and a glass of wine in a cozy restaurant."},
            {"role": "user", "content": "Write a prompt for an image of: " + prompt}            
        ]
    )
    text = response['choices'][0]['message']['content'].strip() #
    return text

# Define the functions that will be used by the agent
functions=[
         
        {
            "name": "generate_image",
            "description": "Use the image prompt you generated from the user's input to generate an image using Stable Diffusion",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image prompt you generated from the user's input",
                    }
                },
                "required": ["prompt"],
            },
        },
    ]
async def octo_gen(prompt):
    '''Generate an image using the prompt'''
    image_response, image = generate_image(prompt)
    return image_response, image

### OPENAI API SETUP ###
settings = {
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}

gpt3 = "gpt-3.5-turbo-0613"
gpt4 = "gpt-4-0613"

ai = AsyncAIChat(api_key="key", 
        system=OCTO_SYSTEM_PROMPT, 
        model=gpt3,
        console=False, 
        params=settings,
        )


@cl.on_chat_start
async def start_chat():

    await cl.Avatar(
        name="stability.ai",
        url="https://avatars.githubusercontent.com/u/100950301?s=200&v=4",
    ).send()
    await cl.Avatar(
        name="leonardo.ai",
        url="https://avatars.githubusercontent.com/u/115455698?s=200&v=4",
    ).send()
    await cl.Avatar(
        name="chatgpt",
        url="https://avatars.githubusercontent.com/u/14957082?s=200&v=4",
    ).send()
    await cl.Avatar(
        name="octoai",
        url="https://avatars.githubusercontent.com/u/53242847?s=200&v=4",
    ).send()

    await cl.Message(
        content="Welcome! Please enter your prompt to generate an image", author="octoai"
    ).send()
#     # send a welcome message to the user and ask them to provide a prompt
#     cl.Message(content="Welcome to the Stable Diffusion Image Prompter. Please provide a prompt for an image you would like to generate.").send()


@cl.on_message
async def main(message: str):     
    await cl.Message(author="octoai", content="").send()

    completions = []

    # response = ai(message, tools=[octo_gen])
    

    # response = ai(message)
    # openai_msg = cl.Message(author="chatgpt", content=response)


    # openai_msg = cl.Message(author="chatgpt", content="")

    # async for chunk in await ai.stream(message):
    #     response_td = chunk["response"]    # dict contains "delta" for the new token and "response"
    #     # print(response_td)
    #     completions.append(response_td)
    #     await openai_msg.stream_token(response_td, is_sequence=True)


    # response = ai.stream(message)

    # async for chunk in response:
    #     response_td = chunk["response"]    # dict contains "delta" for the new token and "response"
    #     # print(response_td)
    #     completions.append(response_td)

    # for token in completions:
    #     openai_msg.stream_token(token)
    
    # await openai_msg.send()

    # cl.Message(author="stability.ai", content=response).send()

    # # send a message informing the user that the agent is generating an image

    # cl.Message(author="stability.ai", content="Now generating the image...").send()

    # Generate an image using the prompt

    # convert completions list to string
    # response = ''.join(completions)

    image_response, image = await octo_gen(message)

    image_name = cl.user_session.get("generated_image")

    if image:
        elements = [
            cl.Image(
            content=image,
            name=image_name,
            display="inline",
            )
        ]
        await cl.Message(author="octoai", content=image_response, elements=elements).send()







    # prompt = sd_prompter(message)

    # send a message informing the user that the agent is creating a prompt
    # cl.Message(author="stability.ai", content="Created a prompt for your image...").send()


    

    # cl.Message(author="stability.ai", content=f"Created prompt: {prompt}").send()

    

    # cl.user_session.set(image_response, image)
    # cl.user_session.set("generated_image", name)

    
        
    
    