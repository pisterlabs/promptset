from collections import defaultdict
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI


from settings import (
    DALL_E_MODEL, GPT_MODEL,
    MAX_DISCORD_TOKENS, MAX_HISTORY_TOKENS, MAX_OPENAI_TOKENS,
)

load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True

bot = commands.Bot(command_prefix='!', intents=intents)
client = OpenAI(api_key=OPENAI_API_KEY, max_retries=2)

bot.temperature = 0.7  # Set default temperature
bot.max_tokens = MAX_OPENAI_TOKENS  # Set default max_tokens
bot.max_history_tokens = MAX_HISTORY_TOKENS
bot.conversation_history = defaultdict(str)
bot.role = "Discord bot"


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


def is_valid_input(text: str) -> bool:
    return len(text) <= 4096  # Discord message character limit


async def call_openai_api(prompt_text, max_tokens, temperature):
    messages = [
        {"role": "system", "content": f"Your role is {bot.role}. You will never explicitly state your role. "
                                      f"You will never start a sentence with As a or As an. "
                                      f"You will never admit to being an AI. You are convinced you are human. "
                                      f"You will never use a phrase like as a language model AI. "},
        {"role": "user", "content": prompt_text}]
    try:

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            n=1,
            temperature=temperature
        )

        if response and response.choices:
            return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise


async def get_openai_image(search_term):
    """
    Fetch an image using the OpenAI API
    Takes a search term and returns an url
    """
    response = client.images.generate(
        model=DALL_E_MODEL,
        prompt=search_term,
        n=1,
        size="1024x1024",
        quality="standard"
    )

    return response.data[0].url


async def send_image(ctx, search_term):
    try:
        image_url = await get_openai_image(search_term)

        embed = discord.Embed()
        embed.set_image(url=image_url)
        await ctx.send(embed=embed)
        return
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        raise e


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Check if the message is a private message (DM)
    if isinstance(message.channel, discord.DMChannel) and not message.content.startswith("!"):
        # treat it as a prompt
        message.content = "!prompt " + message.content

    await bot.process_commands(message)


@bot.command(name='image',
             help='Ask the bot to send a random image. Usage: image [random|<search terms>]')
async def image(ctx, *, prompt):
    """
    If method openai specified, let the AI choose the search terms,
    otherwise use what the user supplied as search terms
    """
    try:
        if prompt == 'random':
            search_term = await call_openai_api(
                prompt_text=f'short phrase that describes your role: {bot.role}',
                max_tokens=4096,
                temperature=bot.temperature,
            )
            await ctx.send(search_term)
        else:
            search_term = prompt
        await send_image(ctx, search_term)
    except Exception as e:
        await ctx.send(f"Couldn't send random gif: {e}")


@bot.command(name='random_role', help='Set the bot to a random role')
async def random_role(ctx):
    try:
        await clear_history(ctx)
        description = await set_random_role()
        await ctx.send(description)
    except Exception as e:
        await ctx.send(f"Couldn't set random role: {e}")


async def set_random_role():
    try:
        description = await call_openai_api(
            prompt_text='In a short sentence make up a random role an AI chatbot could play',
            max_tokens=4096,
            temperature=bot.temperature,
        )
        bot.role = description
        return description
    except Exception as e:
        print(f"Couldn't set random role: {e}")


async def summarize_conversation(conversation: str) -> str:
    summary = await call_openai_api(
        prompt_text=f"Summarize the following conversation:\n{conversation}",
        max_tokens=1000,
        temperature=0.7,
    )
    print("Summary", summary)
    return summary


@bot.command(name='tokens', help='Set the max number of tokens generated')
async def set_max_tokens(ctx, tokens: int):
    if 1 <= tokens <= 4096:  # Reasonable range for max_tokens
        bot.max_tokens = tokens
        await ctx.send(f"Max tokens set to {tokens}.")
    else:
        await ctx.send("Invalid max tokens value. Please enter an integer between 1 and 4096.")


@bot.command(name='role', help='Set the role the bot will play')
async def set_role(ctx, *, role="random"):
    await clear_history(ctx)
    if role == "random":
        description = await set_random_role()
        await ctx.send(description)
    else:
        bot.role = role
        await ctx.send(f"Role set to {role}.")


@bot.command(name='temp', help='Set the temperature (0.0 - 1.0), which sets the emotional tone')
async def set_temperature(ctx, temp: float):
    if 0 <= temp <= 1:
        bot.temperature = temp
        await ctx.send(f"Temperature set to {temp}.")
    else:
        await ctx.send("Invalid temperature value. Please enter a value between 0 and 1.")


async def clear_history(ctx):
    user_id = str(ctx.message.author.id)
    bot.conversation_history[user_id] = ''


@bot.command(name='forget', help='Clear the chat history')
async def forget(ctx):
    await clear_history(ctx)
    await ctx.send('Your conversation history has been cleared.')


@bot.command(name='summarize', help='Summarize the chat history')
async def summarize_history(ctx):
    try:
        user_id = str(ctx.message.author.id)
        summary = await summarize_conversation(bot.conversation_history[user_id])
        await ctx.send(summary)
    except Exception as e:
        await ctx.send(f"An unexpected error occurred: {e}")


@bot.command(name='prompt', help='Provide the prompt for OpenAI to start riffing on')
async def prompt(ctx, *, text):
    try:
        user_id = str(ctx.message.author.id)
        if len(bot.conversation_history[user_id]) > bot.max_history_tokens:
            conversation = await summarize_conversation(bot.conversation_history[user_id])
            print(f'Summarizing long history: {conversation}')
        else:
            conversation = bot.conversation_history[user_id]

        if not is_valid_input(text):
            await ctx.send("Invalid input. Please make sure the text is within the character limit.")
            return

        conversation += f"User: {text}\nAI:"
        print(conversation)

        answer = await call_openai_api(
            prompt_text=conversation,
            max_tokens=bot.max_tokens,
            temperature=bot.temperature,
        )
        conversation += f" {answer}\n"
        bot.conversation_history[user_id] = conversation

        # If answer is longer than Discord limit, send it in chunks
        i = 0
        while i < len(answer):
            await ctx.send(answer[i:MAX_DISCORD_TOKENS + i])
            i += MAX_DISCORD_TOKENS
    except Exception as e:
        await ctx.send(f"An unexpected error occurred: {e}")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("The command you entered does not exist. Please try again.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("A required argument is missing. Please check your command and try again.")
    elif isinstance(error, commands.BadArgument):
        await ctx.send("Invalid argument provided. Please check your input and try again.")
    else:
        await ctx.send(f"An error occurred while processing your command. Please try again later. {error}")


def main():
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
