import discord
import os  # default module
from dotenv import load_dotenv
from discord.ext import commands
import requests
import numpy as np
import cohere

API_KEY = "u4uSutHawHYkDkfWHZ0TL0ETVmE1G6lGrLlFYnHW"

co = cohere.Client('u4uSutHawHYkDkfWHZ0TL0ETVmE1G6lGrLlFYnHW')
BASE_MESSAGE_URL = "https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def get_embeddings(text):
    embeddings = co.embed(text).embeddings
    return embeddings


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


load_dotenv()  # load all the variables from the env file
bot = discord.Bot()


@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")


@bot.slash_command(name="get_messages", description="Get the last 100 messages in the current channel")
async def get_messages(ctx: commands.Context):
    channel = ctx.channel
    messages = []
    async for message in channel.history(limit=100):
        messages.append(message.content)

    messages_str = "\n".join(messages)
    await ctx.send(f"Recent messages:\n{messages_str}")


@bot.command()
async def embed(ctx):
    embed = discord.Embed(
        title="My Amazing Embed",
        description="Embeds are super easy, barely an inconvenience.",
        # Pycord provides a class with default colors you can choose from
        color=discord.Colour.blurple(),
    )
    embed.add_field(name="A Normal Field",
                    value="A really nice field with some information. **The description as well as the fields support markdown!**")

    embed.add_field(name="Inline Field 1", value="Inline Field 1", inline=True)
    embed.add_field(name="Inline Field 2", value="Inline Field 2", inline=True)
    embed.add_field(name="Inline Field 3", value="Inline Field 3", inline=True)

    # footers can have icons too
    embed.set_footer(text="Footer! No markdown here.")
    embed.set_author(name="Pycord Team",
                     icon_url="https://example.com/link-to-my-image.png")
    embed.set_thumbnail(url="https://example.com/link-to-my-thumbnail.png")
    embed.set_image(url="https://example.com/link-to-my-banner.png")

    # Send the embed with some text
    await ctx.respond("Hello! Here's a cool embed.", embed=embed)


@bot.slash_command(name="search3", description="Search for messages containing a keyword")
async def search_messages(ctx: commands.Context, keyword: str):
    channel = ctx.channel
    messages = []
    async for message in channel.history(limit=100):
        if keyword in message.content:
            messages.append(f"{message.content}\nURL: {message.jump_url}")

    if messages:
        messages_str = "\n".join(messages)

        # 將訊息分割成多個部分，每個部分不超過 1000 字元
        message_parts = [messages_str[i:i+1000]
                         for i in range(0, len(messages_str), 1000)]
        for part in message_parts:
            await ctx.send(f"Matching messages:\n{part}")
    else:
        await ctx.send(f"No messages found containing '{keyword}'")


@bot.slash_command(name="search_embedding", description="Search for messages by embedding")
async def search(ctx, query: str):
    search_results = []
    async for msg in ctx.channel.history(limit=100):
        embeddings1 = co.embed([query]).embeddings[0]
        embeddings2 = co.embed([msg.content]).embeddings[0]
        similarity = cosine_similarity(embeddings1, embeddings2)
        if similarity > 0.5:
            search_results.append(msg)
            print(search_results)

    if len(search_results) > 0:
        result_str = "Search results:\n\n"
        for result in search_results:
            message = result
            result_str += f"{message.author}: [{message.content}]({message.jump_url})\n"

        await ctx.send(content=result_str)
    else:
        await ctx.send("No matching messages found.")


bot.run(
    os.environ['TOKEN'])
