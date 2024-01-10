# Imports
import os
import openai
import sys
import argparse
import requests
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.llms.openai import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import io
from PIL import Image

# Parameters
GPT_model = ""

# Key setup
def setup_openai(model = "gpt-3.5-turbo", serper_key = False):
    global GPT_model
    GPT_model = model
    # Setup OpenAI API
    local_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_key = os.path.join(local_dir, 'openai_key.txt')

    try:
        with open(path_to_key, 'r') as file:
            openai_key = file.read()
    except Exception as e:
            print(e)

    openai.api_key = openai_key
    os.environ['OPENAI_API_KEY'] = openai_key

    if serper_key:
        path_to_key = os.path.join(local_dir, 'serper_key.txt')
        try:
            with open(path_to_key, 'r') as file:
                serper_key = file.read()
        except Exception as e:
            print(e)
        os.environ["SERPER_API_KEY"] = serper_key
        return openai_key, serper_key

    return openai_key

# Functions - missing TTS and STT
def text_generator(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_response_text(model, temperature, prompt, program, max_tokens)
    return response

def text_feedback(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_feedback_text(model, temperature, prompt, program, max_tokens)
    return response

def image_generator(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_response_image(prompt, quality)
    return response

def image_feedback(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_feedback_image(temperature, screenshot, program, max_tokens)
    return response

def code_generator(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_response_code(model, temperature, prompt, program, max_tokens)
    return response

def code_feedback(model, temperature, prompt, program, screenshot, max_tokens, quality):
    response = generate_feedback_code(model, temperature, prompt, program, max_tokens)
    return response

def shortcuts(model, temperature, prompt, program, screenshot, max_tokens, quality):
    shortcuts = generate_response_shortcuts(model, temperature, prompt, program, max_tokens)
    return shortcuts

def use_cases(model, temperature, prompt, program, screenshot, max_tokens, quality):
    use_cases = generate_response_usecases(model, temperature, prompt, program, max_tokens)
    return use_cases



# Text to text queries
def generate_response_text(model, temperature, prompt, program, max_tokens):
    
    my_key = setup_openai()

    messag=[{"role": "system", "content": "You are a text generator bot. From the initial text or keywords, \
             generate an abstract for a bigger project containing what was said in the initial text. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you some key words or a message. with all of them, you will generate a text. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the keywords or message."]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def generate_feedback_text(model, temperature, prompt, program, max_tokens):
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are a feedback bot. From the given text or code, you will generate a feedback. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a paragraph with what i want to give feedback to. write the feedback and nothing else. \
                    give feedback on the ideas, the writing style, the grammar... \
                    if its a poem, take into account the rhymes, the rhythm, the metaphors... \
                    if its a story, take into account the plot, the characters, the setting... \
                    and so on."]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the text or code to give feedback to. "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def generate_response_code(model, temperature, prompt, program, max_tokens):
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are a code generator bot. From the given text or keywords, \
             generate a code snippet that fullfills the given task. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you some key words or a message. with all of them, you will generate a code snippet. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the keywords or message."]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def generate_feedback_code(model, temperature, prompt, program, max_tokens):
    my_key = setup_openai()
    messag=[{"role": "system", "content": "You are a feedback bot. From the given text or code, you will generate a feedback. "}]
    
    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you a paragraph with what i want to give feedback to. write the feedback and nothing else. \
                    give feedback to the code itself, \
                    or to the expected output. look out for errors, mistyped things... \
                    also, take into account the readability of the code, the comments, the variable names... "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the text or code to give feedback to. "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )


    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def generate_response_shortcuts(model, temperature, prompt, program, max_tokens):
    my_key = setup_openai()
    
    messag=[{"role": "system", "content": "You are a shortcuts bot for windows laptops. From the given program, \
             give a list of 10 of the most used shortcuts and the commands they execute and nothing else \
             The format is: the name as short as possible, :, the shortcut "}]

    # User history to condition the bot - how do we like the answers to be?
    history_user = ["i'll give you the name of a program. write the list of 10 shortcuts for it, nothing else. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the program. ill say nothing else than the shortcuts "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(program)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result

def generate_response_usecases(model, temperature, prompt, program, max_tokens):
    my_key = setup_openai()
    
    messag=[{"role": "system", "content": "You are a usecases bot. For the given program, \
             you will generate up to ten usecases, and nothing else. \
             The format is: the name as short as possible, :, the usecase "}]
    
    # User history to condition the bot - how do we like the answers to be?         
    history_user = ["i'll give you a program. write up to then usecases for it, nothing else. "]

    # Chat history to condition the bot
    history_bot = ["Yes, I'm ready! Please provide the program name. ill give nothing more than the answer "]

    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(program)})

    response = openai.chat.completions.create(
        model=model,
        messages=messag,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = ''
    for choice in response.choices:
        result += choice.message.content

    return result


def generate_response_image(prompt, quality='high'):
    # Generate reference image from prompt
    my_key = setup_openai()

    if quality == 'low':
        size = "256x256"
    elif quality == 'medium':
        size = "512x512"
    elif quality == 'high':
        size = "1024x1024"

    # elif
    # we don't have that much money

    # Generate a nice image prompt from the text/keywords
    messag=[{"role": "system", "content": "You are a keywords to image prompt generator.\
              From the given keywords or message, you will generate a \
             prompt for Dalle to generate a nice reference image. "}]
    history_user = ["i'll give you some key words or a message. with all of them, you will generate a nice reference image. "]
    history_bot = ["Yes, I'm ready! Please provide the keywords or message."]
    for user_message, bot_message in zip(history_user, history_bot):
        messag.append({"role": "user", "content": str(user_message)})
        messag.append({"role": "system", "content": str(bot_message)})
    messag.append({"role": "user", "content": str(prompt)})
    response = openai.chat.completions.create(
        model=GPT_model,
        messages=messag,
        max_tokens=200,
        temperature=0.8,
    )
    result = ''
    for choice in response.choices:
        result += choice.message.content

    # Generate image from prompt
    response = openai.images.generate(
    prompt=result,
    n=1,
    size=size
    )
    image_url = response.data[0].url

    # Get the image from the url
    response = requests.get(image_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Convert the image content to BMP format
        image = Image.open(io.BytesIO(response.content))
    else:
        image = None

    return image

def generate_feedback_image(temperature, image, program, max_tokens):
    my_key = setup_openai()
    
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {my_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "You are a feedback bot. From the given image, you will generate a feedback."
            },
            {
            "type": "text",
            "text": f"i'll give you an image from the program {program} with what i want to give feedback to. write the feedback and nothing else. take into account the artistic style, the composition, the colors..., if it's a slide for a presentation take readibility into account, and the amount of text. if it's a logo, take into account the colors, the shapes, the font... if it's a graph, take into account the colors, the labels, the axis... and so on"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image}"
            }
            }
        ]
        }
    ],
    "max_tokens": max_tokens,
    "temperature": temperature
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

    assistant_message = response['choices'][0]['message']['content']

    return assistant_message
