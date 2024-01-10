import html
import asyncio
import os
import re
import time
import openai
import discord
import logging
from discord.ext import commands
import configparser
import requests
import json

# Load the config file
config = configparser.ConfigParser()
config.read("config.ini")

# Set up OpenAI API key
try:
    openai.api_key = config["OpenAI"]["api_key"]
except openai.OpenAIError as e:
    raise Exception(f"Error setting OpenAI API key: {e}")

# Set up the logging system
logging.basicConfig(filename='bot.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Create the bot instance
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='!', intents=intents)

# List to store the questions in a queue
question_queue = []

# Dictionary to store user IDs and the time they last asked a question
rate_limit_dict = {}

# List to store the image requests in a queue
image_queue = []

# Load the list of bad words from the file
with open("badwords.txt") as f:
    bad_words = f.read().splitlines()

def sanitize_input(user_input):
    sanitized_input = user_input.replace("../", "").replace("..\\", "")
    return sanitized_input

	
# Coroutine to handle sending the response to the user
async def send_message(channel, response, message):
    print("Sending message...")
    if len(response) > 2000:
        await channel.send(f"{message.author.mention}, My response is too long, sending in a Direct Message.")
        chunks = [response[i:i + 2000] for i in range(0, len(response), 2000)]
        for chunk in chunks:
            await message.author.send(chunk)
    else:
        await channel.send(f"{message.author.mention}, {response}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = message.author.id

    if message.content.startswith("!chat"):
        question = message.content[5:].strip()
        sanitized_question = sanitize_input(question)

        # Escape HTML characters
        question = html.escape(question)

        if user_id not in rate_limit_dict:
            rate_limit_dict[user_id] = time.time()
        else:
            time_since_last_question = time.time() - rate_limit_dict[user_id]
            if time_since_last_question < 3:
                await message.channel.send(f"{message.author.mention}, you are asking too fast! Please slow down.")
                return

        response = await generate_response(question)
        if response is None:
            return

        await send_message(message.channel, response, message)

    elif message.content.startswith("!image"):
        prompt = message.content[6:].strip()
        sanitized_prompt = sanitize_input(prompt)
        
        # Add the image request to the image_queue
        image_queue.append((message.channel, message, sanitized_prompt))


async def generate_response(question):
    try:
        prompt = (
            "You are a helpful assistant. User: "
            + question
            + "\nAssistant:"
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }

        data = json.dumps({
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": question}],
            "max_tokens": 1024,
            "n": 1,
            "stop": None,
            "temperature": 0.3,
        })

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=data)
        response_json = response.json()
        if "choices" in response_json:
            response_text = response_json["choices"][0]["message"]["content"]
        else:
            logging.error(f"Error in API response: {response_json}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Error answering question: {e}")
        return None
    return response_text.strip()

async def generate_image(prompt):
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai.api_key}",
        }

        data = {
            "prompt": prompt,
            "n": 2,
            "size": "256x256",
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data,
        )

        response_json = response.json()
        if "data" in response_json:
            image_url = response_json["data"][0]["url"]
        elif "error" in response_json:
            error_message = response_json["error"]["message"]
            if "safety system" in error_message:
                return "safety_system_error"
            else:
                print(f"Error in API response: {response_json}")
                image_url = None
        else:
            print(f"Error in API response: {response_json}")
            image_url = None
    except Exception as e:
        print(f"Error generating image: {e}")
        image_url = None

    return image_url

async def handle_question_queue():
    while True:
        if question_queue:
            channel, message, question = question_queue.pop(0)

            response = await generate_response(question)
            if response is None:
                continue

            try:
                await send_message(channel, response, message)
            except discord.errors.Forbidden as e:
                logging.error(f"Error sending message: {e}")
            except Exception as e:
                logging.error(f"Error: {e}")
        await asyncio.sleep(0.1)

async def handle_image_queue():
    while True:
        if image_queue:
            channel, message, prompt = image_queue.pop(0)

            image_url = await generate_image(prompt)

            if image_url == "safety_system_error":
                await channel.send(f"{message.author.mention}, your request was rejected as a result of our safety system.")
            elif image_url is not None:
                # Download the image to a temporary file
                with requests.get(image_url, stream=True) as r:
                    r.raise_for_status()
                    with open("temp_image.jpg", "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                # Upload the image to Discord
                with open("temp_image.jpg", "rb") as f:
                    await channel.send(f"{message.author.mention}, here's the generated image:", file=discord.File(f))
                
                # Remove the temporary file
                os.remove("temp_image.jpg")

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(handle_question_queue())
    loop.create_task(handle_image_queue())  # Add this line
    loop.run_until_complete(bot.start(config["Discord"]["token"]))

