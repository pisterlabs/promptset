# Importing Libraries
import discord
from discord.ext import commands
import asyncio
import yt_dlp
import os
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from api_keys import discord_key, openai_key

intents = discord.Intents.default() # Create intents object
intents.message_content = True # Enable message content intents
bot = commands.Bot(command_prefix='!', intents=intents) # Create a bot instance with a command prefix

key = discord_key # Set your bot token here

voice_clients = {} # Dictionary to store voice clients for each guild
queues = {} # Dictionary to store queues for each guild

# YTDL options for downloading audio from YouTube
yt_dl_opts = {
    'format': 'bestaudio/best',
    'extract_flat': True,
}

ytdl = yt_dlp.YoutubeDL(yt_dl_opts) # Initialize the YoutubeDL object

ffmpeg_executable = "C:\ProgramData\chocolatey\lib/ffmpeg-full/tools/ffmpeg/bin/ffmpeg.exe"

ffmpeg_options = {
    'options': "-vn -b:a 192k",
    'executable': ffmpeg_executable
}



template = """User: {user_input} 
Bot: {bot_response}""" # Template for the prompt

os.environ["OPENAI_API_KEY"] = openai_key # Set your OpenAI API key here

llm = OpenAI() # Initialize the OpenAI language model

# Create the LangChain with the prompt and language model
llm_chain = LLMChain(prompt=PromptTemplate(template=template, input_variables=["user_input", "bot_response"]), llm=llm)

# Dictionary to store user preferences
user_preferences = {}

# Function to update user preferences based on the current input
def update_user_preferences(user_input):
    if "favorite color" in user_input.lower():
        words = user_input.lower().split()
        color_index = words.index("color")
        favorite_color = words[color_index + 2]
        user_preferences["favorite_color"] = favorite_color
        print("Bot: Got it! Your favorite color is now set to", favorite_color)

# Function to interact with the chatbot
def chat_with_bot(user_input):
    inputs = {"user_input": user_input, "bot_response": "", "user_preferences": user_preferences}
    bot_response = llm_chain.run(inputs)
    update_user_preferences(user_input)
    print("Bot:", bot_response)
    return bot_response


# YTDLSource class for streaming audio from YouTube
class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get('title')
        self.url = data.get('url')

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=False):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))
        if 'entries' in data:
            data = data['entries'][0]
        filename = data['url'] if stream else ytdl.prepare_filename(data)
        return cls(discord.FFmpegPCMAudio(filename, **ffmpeg_options), data=data)

@bot.event
async def on_ready():
    print(f"Bot logged in as {bot.user}")

    # Send a welcome message and command summary when the bot is ready
    guild = bot.guilds[0]  # Assuming the bot is in at least one guild
    channel = guild.system_channel  # You can change this to the desired channel

    if channel:
        welcome_message = (
            f"Hi there! I'm {bot.user.name}, your friendly music player bot.\n"
            f"Here's a quick summary of available commands:\n"

            f" - `!talk_ai <message>`: Talk with me about song suggestions or anything else music related.\n"
            f" - `!play <url>`: Play a song in the voice channel (or queue if song is already playing).\n"
            f" - `!skip`: Skip the currently playing song.\n"
            f" - `!pause`: Pause the currently playing song.\n"
            f" - `!resume`: Resume the paused song.\n"
            f" - `!stop`: Stop the currently playing song and disconnect from the voice channel.\n"
        )
        await channel.send(welcome_message)

# Process commands
@bot.event
async def on_message(message):
    await bot.process_commands(message)

# Talk to the AI and get a response
@bot.command(name='talk_ai')
async def talk_ai(ctx, *, user_input):
    try:
        bot_response = chat_with_bot(user_input)
        await ctx.send(f"AI Response: {bot_response}")
    except Exception as err:
        print(f"Error talking to AI: {err}")
        await ctx.send(f"An error occurred while talking to AI: {err}")

# Plays a song in the voice channel or queues a song if one is already playing
@bot.command(name='play')
async def play(ctx, url):
    try:
        if ctx.guild is None:
            await ctx.send("This command can only be used in a server.")
            return

        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send("You are not in a voice channel.")
            return

        voice_channel = ctx.author.voice.channel

        # Check if the bot is already connected to a voice channel in the same guild
        if ctx.guild.id in voice_clients:
            # Bot is already connected, add the song to the queue
            if ctx.guild.id not in queues:
                queues[ctx.guild.id] = []
            queues[ctx.guild.id].append(url)
            await ctx.send(f"Added to queue: {url}")

            # Optionally, you can display the current queue here if needed
            # await display_queue(ctx)

        else:
            # Bot is not connected, connect to the voice channel and play the song
            voice_client = await voice_channel.connect()
            voice_clients[ctx.guild.id] = voice_client

            if ctx.guild.id not in queues:
                queues[ctx.guild.id] = []

            queues[ctx.guild.id].append(url)

            if ctx.voice_client.is_playing() or ctx.voice_client.is_paused():
                await ctx.send(f"Added to queue: {url}")
            else:
                await play_next(ctx)

    except Exception as err:
        print(f"Error playing song: {err}")
        await ctx.send(f"An error occurred while playing the song: {err}")

# Plays the next song in the queue
async def play_next(ctx):
    try:
        if queues[ctx.guild.id]:
            url = queues[ctx.guild.id].pop(0)
            player = await YTDLSource.from_url(url, loop=bot.loop)
            ctx.voice_client.play(player, after=lambda e: print('Player error: %s' % e) if e else None)
            await ctx.send(f"Now playing: {player.title}")
        else:
            await ctx.voice_client.disconnect()
            del voice_clients[ctx.guild.id]
    except Exception as err:
        print(f"Error playing next song: {err}")

# Skips the currently playing song
@bot.command(name='skip')
async def skip(ctx):
    try:
        if ctx.guild.id in queues and len(queues[ctx.guild.id]) > 0:
            ctx.voice_client.stop()
            await play_next(ctx)
        else:
            await ctx.send("No more songs to skip.")
    except Exception as err:
        print(f"Error skipping song: {err}")

# Pauses the currently playing song
@bot.command(name='pause')
async def pause(ctx):
    try:
        if ctx.guild.id in voice_clients:
            ctx.voice_client.pause()
    except Exception as err:
        print(f"Error pausing song: {err}")

# Resumes the currently paused song
@bot.command(name='resume')
async def resume(ctx):
    try:
        if ctx.guild.id in voice_clients:
            ctx.voice_client.resume()
    except Exception as err:
        print(f"Error resuming song: {err}")


# Stops any song the bot is currently playing and disconnects from the voice channel
@bot.command(name='stop')
async def stop(ctx):
    try:
        if ctx.guild.id in voice_clients:
            ctx.voice_client.stop()
            await ctx.voice_client.disconnect()
            del voice_clients[ctx.guild.id]
            if ctx.guild.id in queues:
                del queues[ctx.guild.id]
    except Exception as err:
        print(f"Error stopping song: {err}")

bot.run(key)
