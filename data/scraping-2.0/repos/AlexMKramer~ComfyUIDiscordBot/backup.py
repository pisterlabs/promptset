import asyncio
import json
import queue
import websocket
import random
import discord
from discord.ext import commands
from discord import option
import os
from PIL import Image
import io
import tempfile
import comfyAPI
from typing import Optional
from dotenv import load_dotenv
import re
from lyricsgenius import Genius
import openai

load_dotenv()
TOKEN = os.getenv('TOKEN')
folder_path = os.getenv('FOLDER_PATH')
base_model = 'sd_xl_base_1.0_0.9vae.safetensors'
refiner_model = 'sd_xl_refiner_1.0_0.9vae.safetensors'
genius_token = os.getenv('GENIUS_TOKEN')
genius = Genius(genius_token)
openai.api_key = os.getenv('OPENAI_API_KEY')
prompt = comfyAPI.prompt
intents = discord.Intents.default()
bot = commands.Bot(command_prefix='/', intents=intents)
bot.auto_sync_commands = True

#  Height and Width options
height_width_option = [
    "1024 1024",
    "1152 896",
    "896 1152",
    "1216 832",
    "832 1216",
    "1344 768",
    "768 1344",
    "1536 640",
    "640 1536"
]

#  Style Json parse
with open("resources/sdxl_styles.json", 'r') as sdxl_styles:
    data = json.load(sdxl_styles)

# Parse Style names from sd_xl_styles.json
style_names = [entry["name"] for entry in data]

#  Prompts Json parse
with open("resources/prompts.json", 'r') as sdxl_prompts:
    prompts_data = json.load(sdxl_prompts)

example_subjects = prompts_data["prompts"]["subjects"]
example_verbs = prompts_data["prompts"]["verbs"]
example_locations = prompts_data["prompts"]["locations"]

initial_prompt = [{'role': 'user',
                   'content': "Using song lyrics, come up with a prompt for an image generator.  "
                              "Please follow the format exactly. The format should be broken down "
                              "like this: {Art Style}, {Subject}, {Details}, {Color}\n The art style "
                              "should be determined by the overall impression of the song.  If it is "
                              "sad, then something like La Douleur should be used. If it is happy, "
                              "perhaps a vibrant street art style.\nThe Subject should be determined "
                              "by who the song is about.  If the song is about a couple trying to "
                              "escape the city, then the subject should be a couple.\nThe Details "
                              "should be determined by descriptive words used in the song.  If they "
                              "mention empty bottles, then add empty bottles to the prompt.\nThe "
                              "color should be determined by the mood of the song.  If the mood is a "
                              "happy one, use bright colors.\nHere is an example:\n{A dreamlike and "
                              "ethereal art style}, {a couple standing on a cliffside embracing, "
                              "overlooking a surreal and beautiful landscape}, {sunset, grassy, "
                              "soft wind}, {soft pastels, with hints of warm oranges and pinks}"},
                  {'role': 'assistant',
                   'content': "{Vibrant and energetic street art style}, {a group of friends dancing and "
                              "celebrating under the city lights}, {joyful, urban, rhythm}, {bold and lively "
                              "colors, with splashes of neon blues and pinks}"}, ]


async def style_autocomplete(ctx: discord.AutocompleteContext):
    return [name for name in style_names if name.startswith(ctx.value.lower())]


async def height_width_autocomplete(ctx: discord.AutocompleteContext):
    return [height_width for height_width in height_width_option]


def remove_text_before_and_after_braces(text):
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        new_text = text[start_index:end_index + 1]
        return new_text
    else:
        return text


# Find Loras in ComfyUI/models folder and create a list for autocomplete
async def loras_autocomplete(ctx: discord.AutocompleteContext):
    subfolder_name = 'loras'
    # Walk through the directory tree rooted at root_folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            return sorted(
                [os.path.splitext(loras)[0] for loras in subfolder_files if loras.startswith(ctx.value.lower())])

    # If the target subfolder is not found
    return []


async def models_autocomplete(ctx: discord.AutocompleteContext):
    subfolder_name = 'checkpoints'
    # Walk through the directory tree rooted at root_folder
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Check if the target subfolder is in the current directory
        if subfolder_name in dirnames:
            subfolder_path = os.path.join(dirpath, subfolder_name)

            # List files within the target subfolder
            subfolder_files = [file for file in os.listdir(subfolder_path)]
            matching_files = [models for models in subfolder_files if models.startswith(ctx.value.lower())]
            return sorted([matching_files])

    # If the target subfolder is not found
    return []


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')


@bot.slash_command(description='Generate images based on song lyrics!')
@option(
    "song_name",
    description="Enter the song name",
    required=True
)
@option(
    "artist_name",
    description="Enter the artist name",
    required=True
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def interpret(ctx, song_name: str, artist_name: str, model_name: str):
    await ctx.respond(
        f"Generating images for {ctx.author.mention}\n**Song:** {song_name}\n**Artist:** {artist_name}")
    try:
        artist = genius.search_artist(artist_name, max_songs=0, sort="title")
        song = genius.search_song(song_name, artist.name)
    except Exception as e:
        print("Error:", e)
        await ctx.send(f"Unable to find song/artist. Check your spelling and try again.")
        exit(1)

    with open('lyrics.txt', 'w') as f:
        f.write(song.lyrics)

    def extract_text_between_keywords(text, keyword1, keyword2_pattern):
        start_index = text.find(keyword1)
        end_match = re.search(keyword2_pattern, text[start_index:])

        if start_index != -1 and end_match:
            end_index = start_index + end_match.start()
            return text[start_index + len(keyword1):end_index].strip()
        else:
            return ""

    def remove_brackets(text):
        return re.sub(r'\[.*?\]', '', text)

    def remove_quotes(text):
        return text.replace('"', '')

    # Read the text file
    with open('lyrics.txt', 'r') as file:
        file_contents = file.read()

    # Define keywords and pattern for keyword2
    keyword1 = "Lyrics"
    keyword2_pattern = r"\d*Embed|Embed"  # Regular expression to match one or more digits followed by "Embed"

    # Extract the desired text
    extracted_text = extract_text_between_keywords(file_contents, keyword1, keyword2_pattern)

    # Remove the ad in the lyrics if there is one.
    ad_pattern = r'See .*? LiveGet tickets as low as \$\d+You might also like'
    extracted_text = re.sub(ad_pattern, '', extracted_text)

    # Remove the number at the end
    extracted_text = re.sub(r'\d+$', '', extracted_text)

    # Remove anything in brackets
    extracted_text = remove_brackets(extracted_text)

    # Remove quotes
    extracted_text = remove_quotes(extracted_text)

    with open('lyrics.txt', 'w') as f:
        f.write(extracted_text)

    new_prompt = ({"role": "user", "content": "Here are the lyrics I would like in this format:" + extracted_text})
    send_prompt = initial_prompt + [new_prompt]
    print(f'Prompt: {send_prompt}')
    # OpenAI Completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=send_prompt
    )
    reply_content = completion.choices[0].message.content

    print(f'OpenAI reply: {reply_content}')

    new_prompt = remove_text_before_and_after_braces(reply_content)
    prompt["146"]["inputs"]["text_positive"] = new_prompt
    prompt["146"]["inputs"]["text_negative"] = 'disfigured, ugly, disfigured, gross, nsfw, writing'
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)
    prompt["23"]["inputs"]["noise_seed"] = int(seed)
    prompt["146"]["inputs"]["style"] = 'base'
    prompt["5"]["inputs"]["height"] = 1024
    prompt["5"]["inputs"]["width"] = 1024
    if model_name is not None:
        prompt["10"]["inputs"]["ckpt_name"] = model_name
    else:
        prompt["10"]["inputs"]["ckpt_name"] = base_model
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        await ctx.send(
            f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n **Song:** {song_name}\n**Artist:** {artist_name}",
            files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


@bot.event
async def on_connect():
    if bot.auto_sync_commands:
        await bot.sync_commands()
    print(f'Logged in as {bot.user.name}')


@bot.slash_command(description='Generate images based on song lyrics!')
@option(
    "song_name",
    description="Enter the song name",
    required=True
)
@option(
    "artist_name",
    description="Enter the artist name",
    required=True
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def music(ctx, song_name: str, artist_name: str, model_name: str):
    await ctx.respond(
        f"Generating images for {ctx.author.mention}\n**Song:** {song_name}\n**Artist:** {artist_name}")
    try:
        artist = genius.search_artist(artist_name, max_songs=0, sort="title")
        song = genius.search_song(song_name, artist.name)
    except Exception as e:
        print("Error:", e)
        await ctx.send(f"Unable to find song/artist. Check your spelling and try again.")
        exit(1)

    with open('lyrics.txt', 'w') as f:
        f.write(song.lyrics)

    def extract_text_between_keywords(text, keyword1, keyword2_pattern):
        start_index = text.find(keyword1)
        end_match = re.search(keyword2_pattern, text[start_index:])

        if start_index != -1 and end_match:
            end_index = start_index + end_match.start()
            return text[start_index + len(keyword1):end_index].strip()
        else:
            return ""

    def remove_brackets(text):
        return re.sub(r'\[.*?\]', '', text)

    def remove_quotes(text):
        return text.replace('"', '')

    # Read the text file
    with open('lyrics.txt', 'r') as file:
        file_contents = file.read()

    # Define keywords and pattern for keyword2
    keyword1 = "Lyrics"
    keyword2_pattern = r"\d*Embed|Embed"  # Regular expression to match one or more digits followed by "Embed"

    # Extract the desired text
    extracted_text = extract_text_between_keywords(file_contents, keyword1, keyword2_pattern)

    # Remove the ad in the lyrics if there is one.
    ad_pattern = r'See .*? LiveGet tickets as low as \$\d+You might also like'
    extracted_text = re.sub(ad_pattern, '', extracted_text)

    # Remove the number at the end
    extracted_text = re.sub(r'\d+$', '', extracted_text)

    # Remove anything in brackets
    extracted_text = remove_brackets(extracted_text)

    # Remove quotes
    extracted_text = remove_quotes(extracted_text)

    # Split the extracted text into lines
    lines = extracted_text.split('\n')

    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    # Remove lines containing brackets
    lines = [line for line in lines if '[' not in line and ']' not in line]

    # Select 3 random, unique lines
    random_lines = random.sample(lines, min(3, len(lines)))  # Safely sample up to 3 lines

    with open('lyrics.txt', 'w') as f:
        f.write(extracted_text)

    output_line = ', '.join(random_lines)
    new_prompt = song_name + ", " + output_line + ", " + artist_name
    prompt["146"]["inputs"]["text_positive"] = new_prompt
    prompt["146"]["inputs"]["text_negative"] = 'text, words, letters, numbers'
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)
    prompt["23"]["inputs"]["noise_seed"] = int(seed)
    prompt["146"]["inputs"]["style"] = 'base'
    prompt["5"]["inputs"]["height"] = 1024
    prompt["5"]["inputs"]["width"] = 1024
    if model_name is not None:
        prompt["10"]["inputs"]["ckpt_name"] = model_name
    else:
        prompt["10"]["inputs"]["ckpt_name"] = base_model
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        await ctx.send(
            f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n **Song:** {song_name}\n**Artist:** {artist_name}",
            files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


@bot.slash_command(description='Generate random images with a random style')
async def crazy(ctx):
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)  # set seed for base model
    prompt["23"]["inputs"]["noise_seed"] = int(seed)  # set seed for refiner model

    # Random prompt
    # Random subject
    random_subject = random.choice(example_subjects)
    # Random verb
    random_verb = random.choice(example_verbs)
    # Random location
    random_location = random.choice(example_locations)
    new_prompt = f"{random_subject} {random_verb} {random_location}"
    prompt["146"]["inputs"]["text_positive"] = new_prompt
    prompt["146"]["inputs"]["text_negative"] = ''
    prompt["10"]["inputs"]["ckpt_name"] = base_model

    # Random style
    random_entry = random.choice(data)
    random_style = random_entry["name"]
    prompt["146"]["inputs"]["style"] = random_style
    await ctx.respond(
        f"Generating 'crazy' images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {random_style}")

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        await ctx.send(
            f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {random_style}",
            files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


@bot.slash_command(description='Generate images using only words!')
@option(
    "new_prompt",
    description="Enter the prompt",
    required=True
)
@option(
    "new_style",
    description="Enter the style",
    autocomplete=style_autocomplete,
    required=False
)
@option(
    "new_height_width",
    description="Choose the height and width",
    autocomplete=height_width_autocomplete,
    required=False
)
@option(
    "new_lora",
    description="Choose the Lora model",
    autocomplete=loras_autocomplete,
    required=False
)
@option(
    "model_name",
    description="Enter the model name",
    autocomplete=models_autocomplete,
    required=False
)
async def draw(ctx, new_prompt: str, new_style: str, new_height_width: str, new_lora: str, model_name: str):
    if new_lora is not None:
        new_prompt = " <lora:" + new_lora + ":0.5>, " + new_prompt
    if new_style is not None and new_height_width is not None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {new_style}\n**Height/Width:** {new_height_width}")
    elif new_style is not None and new_height_width is None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Style:** {new_style}")
    elif new_style is None and new_height_width is not None:
        await ctx.respond(
            f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}\n**Height/Width:** {new_height_width}")
    else:
        await ctx.respond(f"Generating images for {ctx.author.mention}\n**Prompt:** {new_prompt}")
    prompt["146"]["inputs"]["text_positive"] = new_prompt
    prompt["146"]["inputs"]["text_negative"] = ''
    seed = random.randint(0, 0xffffffffff)
    prompt["22"]["inputs"]["noise_seed"] = int(seed)
    prompt["23"]["inputs"]["noise_seed"] = int(seed)
    if new_style is not None:
        if new_style == 'random':
            random_entry = random.choice(data)
            new_style = random_entry["name"]
        prompt["146"]["inputs"]["style"] = new_style
    else:
        prompt["146"]["inputs"]["style"] = 'base'
    if new_height_width:
        height, width = new_height_width.split()
        prompt["5"]["inputs"]["height"] = int(height)
        prompt["5"]["inputs"]["width"] = int(width)
    else:
        prompt["5"]["inputs"]["height"] = 1024
        prompt["5"]["inputs"]["width"] = 1024
    if model_name is not None:
        prompt["10"]["inputs"]["ckpt_name"] = model_name
    else:
        prompt["10"]["inputs"]["ckpt_name"] = base_model
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(comfyAPI.server_address, comfyAPI.client_id))
    print("Current seed:", seed)
    print("Current prompt:", new_prompt)
    images = comfyAPI.get_images(ws, prompt)
    file_paths = []
    for node_id in images:
        for image_data in images[node_id]:
            image = Image.open(io.BytesIO(image_data))
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                image.save(temp_file.name)
                file_paths.append(temp_file.name)
        file_list = [discord.File(file_path) for file_path in file_paths]
        if new_style is not None and new_height_width is not None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {new_style}\n**Height/Width:** {new_height_width}",
                files=file_list)
        elif new_style is not None and new_height_width is None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Style:** {new_style}",
                files=file_list)
        elif new_style is None and new_height_width is not None:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}\n**Height/Width:** {new_height_width}",
                files=file_list)
        else:
            await ctx.send(
                f"Here you are {ctx.author.mention}!\n**Prompt:** {new_prompt}",
                files=file_list)
        for file_path in file_paths:
            os.remove(file_path)


bot.run(TOKEN)
