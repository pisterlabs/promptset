import os
from dotenv import load_dotenv
import discord_bot
import discord
from discord.ext import commands

from openai_schedule_assistant import *
from main import *


load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Create the bot with a command prefix

discord_bot = commands.Bot(command_prefix='!',intents=discord.Intents.default())

# Event handler for when the bot is ready
@discord_bot.event
async def on_ready():
    print(f'Logged in as {discord_bot.user.name} (ID: {discord_bot.user.id})')
    print('------')
    activity = discord.Game(name="Scheduling your events | !help")
    await discord_bot.change_presence(status=discord.Status.online, activity=activity)

@discord_bot.event
async def on_message(ctx):
    try:
        #print(ctx.attachments)
        #if file attachment, download the file and read it. Send the whole text to the AI
        if ctx.attachments:
            for attachment in ctx.attachments:
                await attachment.save(attachment.filename)
                with open(attachment.filename, 'r') as file:
                    data = await message_ai(file.read())
                    await ctx.channel.send(data)
        else:
            if not ctx.author.bot:
                data = await message_ai(ctx.content)
                await ctx.channel.send(data)
        #print(ctx.content)

        #if not ctx.author.bot:
        #    data = message_ai(ctx.content)
        #    await ctx.channel.send(data)


    except Exception as e:
        error_message = f"An error occurred while processing your message: {str(e)}"
        await ctx.channel.send(error_message)


# Initialize and run the bot
if __name__ == "__main__":
    discord_bot.run(TOKEN)
