import io
import datetime
import logging
from urllib.parse import urlparse
from collections import Counter
from tabulate import tabulate
import openai
import discord
from discord.ext import commands
from discord import Intents
import credentials
from sql_connector import DBClient
from helpers import *

logging.getLogger().setLevel(logging.INFO)

openai.api_key = credentials.OPENAI_API_KEY

client = discord.Client(intents=Intents.all())

bot = commands.Bot(intents=Intents.all(), command_prefix='-')

db_client = DBClient()

# Enable this flag if you want to store results in a remote database
USE_DB_FLAG = True

DEBUG = False

FLAGS_PREFIX = "--"

QUALITY_OPTIONS = ["hd", "standard"]
SIZE_OPTIONS = ["portrait", "landscape", "square"]
SIZE_MAPPING = {
    "portrait": "1024x1792",
    "landscape": "1792x1024",
    "square": "1024x1024"
}
MODEL_OPTIONS = ["dall-e-2", "dall-e-3"]
N_OPTIONS = [1, 2, 3, 4]
STYLE_OPTIONS = ["vivid", "natural"]


class GenerateFlags(commands.FlagConverter, delimiter=' ', prefix=FLAGS_PREFIX): # To be integrated later
    quality: str = commands.flag(name='quality', aliases=['q'], default="standard", description=f"quality of the generated image, valid options are: {QUALITY_OPTIONS}")
    size: str = commands.flag(name='size', aliases=['s'], default="square")
    model: str = commands.flag(name='model', aliases=['m'], default="dall-e-3")
    n: str = commands.flag(name='n', default=1)
    style: str = commands.flag(name='style', aliases=['st'], default="vivid")


def process_flags(flags: GenerateFlags):
    # If help flag is present, print help message and exit

    flags.quality = flags.quality.split(" ")[0]
    flags.size = flags.size.split(" ")[0]
    flags.model = flags.model.split(" ")[0]
    flags.n = int(str(flags.n).split(" ")[0])
    flags.style = flags.style.split(" ")[0]

    if flags.quality not in QUALITY_OPTIONS:
        raise commands.BadArgument(f"Invalid quality flag, valid options are: {QUALITY_OPTIONS}")

    if flags.size not in SIZE_OPTIONS:
        raise commands.BadArgument(f"Invalid size flag, valid options are: {SIZE_OPTIONS}")

    if flags.model not in MODEL_OPTIONS:
        raise commands.BadArgument(f"Invalid model flag, valid options are: {MODEL_OPTIONS}")

    if flags.n > max(N_OPTIONS) or flags.n < min(N_OPTIONS):
        raise commands.BadArgument(f"Invalid n flag, n must be between {min(N_OPTIONS)} and {max(N_OPTIONS)}")

    if flags.style not in ["vivid", "natural"]:
        raise commands.BadArgument(f"Invalid style flag, valid options are: ['vivid', 'natural']")

    return flags


def generate_help_message():
    help_message = "Usage: -g, -generate [prompt] [flags]\n"
    help_message += "Flags:\n"
    help_message += f"{FLAGS_PREFIX}q, {FLAGS_PREFIX}quality: quality of the generated image, valid options are: {QUALITY_OPTIONS}\n"
    help_message += f"{FLAGS_PREFIX}s, {FLAGS_PREFIX}size: size of the generated image, valid options are: {SIZE_OPTIONS}\n"
    help_message += f"{FLAGS_PREFIX}m, {FLAGS_PREFIX}model: model to use for generation, valid options are: {MODEL_OPTIONS}\n"
    help_message += f"{FLAGS_PREFIX}n: number of images to generate, valid options are: {N_OPTIONS}\n"
    help_message += f"{FLAGS_PREFIX}st, {FLAGS_PREFIX}style: style of the generated image, valid options are: {STYLE_OPTIONS}\n"
    return help_message


@bot.command(aliases=["g"], help=generate_help_message())
async def generate(ctx, *, flags: GenerateFlags):

    prefix = ctx.bot.command_prefix
    prefix_length = len(prefix)
    prompt = ctx.message.content[prefix_length+2:]

    try:
        process_flags(flags)
    except Exception as error:
        await ctx.reply(error)
        raise

    # Strip flags from message
    for flag in flags:
        flag, value = flag[0], flag[1]
        prompt = prompt.replace(f"{FLAGS_PREFIX}{flag} {value}", "")
        shorthand_flag = flag[0]
        prompt = prompt.replace(f"{FLAGS_PREFIX}{shorthand_flag} {value}", "")

    # Remove extra whitespaces resulting from flag removal
    prompt = " ".join(prompt.split())

    await generate_route(
        ctx.message,
        prompt, 
        model=flags.model,
        quality=flags.quality,
        size=SIZE_MAPPING[flags.size], 
        n=flags.n,
        style=flags.style    
    )


@bot.command(aliases=["v"])
async def variation(ctx):
    await variations_route(ctx.message)


@bot.command(aliases=["d"])
async def dalle(ctx):
    await dalle_route(ctx.message)


def add_prompt(author, prompt, image_urls, timestamp):

    sql = "INSERT INTO dalle(author, prompt, image_urls, timestamp) values (%s, %s, %s, %s);"
    val = (author, prompt, image_urls, timestamp)

    db_client.conn.ping(reconnect=True)

    try:
        cursor = db_client.conn.cursor()
        cursor.execute(sql, val)
        db_client.conn.commit()

    except Exception as err:
        # Refresh connection for next entry
        logging.warning(err)
        db_client.initialize()
        raise


def get_stats():

    stats = {}

    db_client.conn.ping(reconnect=True)

    cursor = db_client.conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM dalle;")

    data = cursor.fetchone()

    stats['runs'] = data[0]

    stats['spent'] = round(15.0/115.0 * stats['runs'], 2)

    cursor.execute(f"SELECT author FROM dalle;")

    data = cursor.fetchall()
    
    user_counts = Counter(data)

    user_data = [] 

    for user, count in user_counts.items():
        val = (user[0], count, f"${round(15.0/115.0 * count, 2)}")
        user_data.append(val)

    user_data.sort(key=lambda x: x[1], reverse=True)

    stats['user_data'] = user_data

    return stats


async def generate_route(
    message: discord.Message, 
    prompt: str,
    model="dall-e-3",
    quality="hd",
    size="1024x1024",
    n=1,
    style="natural"
):
    
    if validate_text(prompt):

        bot_response = await message.reply("Generating...")

        try:
            dalle_response = openai.images.generate(
                model=model,
                prompt=prompt,
                n=n,
                size=size,
                quality=quality,
                style=style,
                response_format="url"
            )
        except Exception as error:
            await bot_response.edit(content=error)
            return error

        cdn_urls = await send_dalle_images(dalle_response, message, prompt)

        revised_prompt = dalle_response.data[0].revised_prompt
        revised_prompt = f"\nRevised your prompt to: {revised_prompt}" if revised_prompt else ""

        await bot_response.edit(content="Done!" + revised_prompt)

        # Add prompt to database
        if not USE_DB_FLAG:
            return "success"

        try:
            add_prompt(message.author, prompt, serialize_image_urls(cdn_urls), message.created_at)
        except Exception as error:
            await bot_response.edit(content="Done, but can't store image in database due to an error." + revised_prompt )
            return error

        return "success"
    
    # If failed first validation or second validation
    await message.reply("Your prompt was flagged by the system.")
    return "failed"


async def variations_route(message: discord.Message):    

    # In case the image is in a reply and not direct attachment
    if message.reference is not None:
        message = await message.channel.fetch_message(message.reference.message_id) 

    # The image is a url not an attachment
    if not message.attachments:
        # Check url is valid
        try: 
            urlparse(message.content)
        except:
            message.reply("Can't download image from URL, url is invalid.")
            return

        image_io = download_image(message.content)
        prompt = f"variation of image {message.content}"

    else:
        image_io = await get_discord_image_from_message(message)
        prompt = f"variation of image {message.attachments[0].filename}"

    try:
        image, resized = await preprocess_input_image(image_io)
    except Exception as error:
        await message.reply("Couldn't process the file, are you sure it's an image?")
        return

    if resized and DEBUG:
        await message.reply(resized)
        await message.channel.send(file=discord.File(fp=image, filename=f"{message.id}.png"))

    try:
        bot_response = await message.reply("Generating...")
        dalle_response = openai.images.create_variation(
            model="dall-e-2",
            image=image.getvalue(),
            n=4,
            size="1024x1024",
            response_format="url"
        )
    except Exception as error:
        await bot_response.edit(content=error)
        return error


    cdn_urls = await send_dalle_images(dalle_response, message, prompt)

    await bot_response.edit(content="Done!")

    if not USE_DB_FLAG:
        return "success"
    # Add prompt to database
    try:
        add_prompt(message.author, prompt, serialize_image_urls(cdn_urls), message.created_at)
    except Exception as error:
        await bot_response.edit(content="Done, but can't store image in database due to an error.")
        return error

    return "success"
    

async def dalle_route(message: discord.Message):

    stats = get_stats()

    stats_text = "```"

    stats_text += f"As of {datetime.date.today().strftime('%b %d, %Y')}, total runs: {stats['runs']}, total spent: ${stats['spent']}\n"

    stats_text += "\nStats breakdown\n"

    stats_text += tabulate(stats['user_data'], headers=('User', '# Runs', 'Spent'), tablefmt='fancy_grid')

    stats_text += "```"

    # await message.reply(stats_text)

    # If reply exceeds 2000 chars, split it into multiple messages each of size 2000 chars
    if len(stats_text) > 2000:
        for i in range(0, len(stats_text), 2000):
            await message.reply(stats_text[i:i+2000])
    else:
        await message.reply(stats_text)


@bot.event
async def on_ready():
    logging.info(f'{bot.user} has connected to Discord!')


bot.run(credentials.DISCORD_TOKEN)