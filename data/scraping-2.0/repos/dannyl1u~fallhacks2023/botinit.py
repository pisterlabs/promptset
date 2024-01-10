import asyncio
import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi
import re
import youtube_dl
# from helper import generate_task_code
import random
import openai
import os
import certifi

# Set the CA certificates path for SSL verification
os.environ["SSL_CERT_FILE"] = certifi.where()


from pydub import AudioSegment

background_tasks = {}
timers = {}
# Create an instance of the bot
# bot = commands.Bot(command_prefix="!")  # Change the prefix as needed
# Specify intents
intents = discord.Intents.all()  # This requests all available events
# If you only need certain events, you can modify this. For example:
# intents = discord.Intents(messages=True, guilds=True) 
audioSong = "audio\\Frog.mp3"
# Create an instance of the bot with intents
bot = commands.Bot(command_prefix="!", intents=intents)
load_dotenv()

def generate_task_code(n=5):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return random.randint(range_start, range_end)

@bot.command()
async def pomodoro(ctx, study_time: float, break_time: float):
    user_id = ctx.author.id
    channel_id = ctx.channel.id
    if user_id in timers:
        timers[user_id]['study_time'] = study_time
        timers[user_id]['break_time'] = break_time
        timers[user_id]['channel_id'] = channel_id
    else:
        timers[user_id] = {
            'study_time': study_time,
            'break_time': break_time,
            'channel_id': channel_id
        }

    await _speak(ctx, f"{ctx.author.mention}, Alright starting your pomodoro timer: {study_time} mins study and {break_time} mins break!")
    await start_timer(ctx, study_time, break_time)


async def start_timer(ctx, study_time, break_time):
    user = ctx.author
    user_id = user.id
    warning_time_fraction = 0.1

    while user_id in timers:  
        # Calculate the time to sleep before sending a warning
        warning_time_study = study_time * warning_time_fraction
        await asyncio.sleep((study_time - warning_time_study) * 60)
        
        if user_id in timers:
            await send_warning(ctx, warning_time_study, break_time)

            # Sleep the remaining study time after the warning
            await asyncio.sleep(warning_time_study * 60)

            channel = bot.get_channel(timers[user_id]['channel_id'])
            await _speak(ctx, f"{user.mention}, your study time is up! Take a break for {break_time} mins!")

            # Calculate the time to sleep before sending a break warning
            warning_time_break = break_time * warning_time_fraction
            await asyncio.sleep((break_time - warning_time_break) * 60)
            
            if user_id in timers:
                await send_warning(ctx, warning_time_study, warning_time_break)

                # Sleep the remaining break time after the warning
                await asyncio.sleep(warning_time_break * 60)

                await _speak(ctx, f"{user.mention}, Break's over! Time to get back to work!")
            else:
                return
        else:
            return


async def send_warning(ctx, remaining_study_time, remaining_break_time):
    user = ctx.author
    user_id = user.id
    if remaining_study_time:
        message = f"{user.mention}, you have {remaining_study_time} mins of study time left!"
    else:
        message = f"{user.mention}, you have {remaining_break_time} mins of break time left!"
    await _speak(ctx, message)



@bot.command()
async def stop(ctx):
    user_id = ctx.author.id
    if user_id in timers:
        del timers[user_id]
        await _speak(ctx, f" grinding has stopped for {ctx.author.mention}")
    else:
        await ctx.send(f"{ctx.author.mention}, you don't have an active pomodoro timer!")


# @bot.event
# async def on_ready():
#     print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
#     channel = bot.get_channel(1162235364719722610)  # Replace YOUR_CHANNEL_ID with the actual channel ID you copied.
#     print('Message sent to the channel')
@bot.command()
async def youtube(ctx, youtube_url: str):
    try:
        # Extract video ID
        pattern1 = r"v=([^&]+)"
        pattern2 = r"(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))([\w-]{11})"
        
        match = re.search(pattern1, youtube_url)
        if not match:
            match = re.search(pattern2, youtube_url)

        if not match:
            raise ValueError("Video ID not found.")

        video_id = match.group(1)

        # Fetch the transcript
        transcript_entries = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_entries])

        # Extract first 50 words (for brevity)
        first_50_words = " ".join(transcript_text.split()[:50])

        # Use OpenAI for summarization
        openai.api_key = os.getenv('OPENAI_API_KEY')
        response = openai.Completion.create(
          engine="davinci",
          prompt=f"The following is the beginning of a youtube video transcript, provide a 1 sentence description of the video:\n\n{first_50_words}",
          max_tokens=50,  # adjust as needed
          temperature=0.75
        )
        print(response)

        summary = response.choices[0].text.strip()

        await ctx.send(f"```Summary:\n{summary}```")
    except Exception as e:
        await ctx.send(f"An error occurred: {e}")


async def on_ready():
    print(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
    channel = bot.get_channel(1162235364719722610)  # Replace YOUR_CHANNEL_ID with the actual channel ID you copied.
    await channel.send("Greetings!!")
    print('Message sent to the channel')

@bot.command()
async def greet_all(ctx):
    for member in ctx.guild.members:
        if not member.bot:  # To ensure you don't send messages to other bots
            if member.name == 'ibuprofen':
                pass
            elif member.name == 'haruuuuu.':
                await ctx.send(f"No greeting for {member.name}")
            elif member.name == 'beansinjeans.':
                await ctx.send(f"Do you smell that {member.name} ?")
            elif member.name == 'ace_2001.':
                await ctx.send(f"Did you hear ibuprofen fart {member.name} ?")
            else:
                await ctx.send(f"Hello, {member.name}!")
        

@bot.command()
async def play(ctx, filename):
    # Check if the user is in a voice channel
    if ctx.author.voice and ctx.author.voice.channel:
        voice_channel = ctx.author.voice.channel
        voice_client = await voice_channel.connect()

        try:
            audio = AudioSegment.from_mp3(filename)  # Load the MP3 file
            audio.export("temp.wav", format="wav")  # Convert to WAV

            voice_client.play(discord.FFmpegPCMAudio("temp.wav"))

            while voice_client.is_playing():
                await asyncio.sleep(1)

            await voice_client.disconnect()
        except Exception as e:
            await ctx.send(f"An error occurred: {str(e)}")
    else:
        await ctx.send("You need to be in a voice channel to use this command.")

@bot.command()
async def join(ctx):
    if ctx.author.voice and ctx.author.voice.channel:
        voice_channel = ctx.author.voice.channel
        voice_client = await voice_channel.connect()
        await ctx.send(f'Joined {voice_channel}')
        
@bot.command()
async def hello(ctx):
    await ctx.send("Hello, I'm your bot!")




async def _remind_countdown(ctx, task_code, seconds):
    username = ctx.author.display_name
    await ctx.send(f"Hey {username}, sleeping for {seconds} seconds...")
    await asyncio.sleep(seconds)
    await _speak(ctx, f"{username}, Failed to complete {task_code}!")
    
    # To prevent keeping references to finished tasks forever,
    # make each task remove its own reference from the dictionary after
    def task_done_callback(task):
        # Do something when task is done.
        print(f"Task {task} completed")
        if task_code in background_tasks:
            background_tasks.pop(task_code)

    background_tasks[task_code].add_done_callback(task_done_callback)

@bot.command()
async def cancel(ctx, task_code):
    background_tasks[task_code].cancel()
    background_tasks.pop(task_code)
    await ctx.send(f"cancelled {task_code}")

@bot.command()
async def completeTask(ctx, task_code):
    if task_code in background_tasks:
        background_tasks[task_code].cancel()
        background_tasks.pop(task_code)
        await ctx.send(f"Congratulations, {ctx.author.name}! You have completed {task_code}!")
    else:
        await ctx.send(f"{task_code} is not in the list of tasks")

@bot.command()
async def tasks(ctx):
    await ctx.send("listing tasks...")
    for task in background_tasks.keys():
        await ctx.send(task)

@bot.command()
async def remind(ctx, task='leetcode', seconds=5):
    task_code = task + "-" + str(generate_task_code())
    await ctx.send(f"Activated reminder for {task_code}!")
    
    _task = asyncio.create_task(_remind_countdown(ctx, task_code, seconds))

    # Storing the task is not always necessary unless you want to access or cancel it later.
    background_tasks[task_code] = _task

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")


async def _speak(ctx, text_to_speak):
    try:
        # Send the TTS message
        await ctx.send(text_to_speak, tts=True)
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

@bot.command()
async def pom_timer(ctx, minutes):
    encouragement = ['keep up the good work!', 'am proud of you', 'lets gooo']
    try:
        minutes = int(minutes)
    except ValueError:
        await ctx.send("Invalid input. Please provide a valid number of minutes.")
        return

    await ctx.send(f"Pomodoro timer started for {minutes} minutes. Work hard!")

    
    await _speak(ctx, encouragement[random.randint(0,3)])

    # Wait for the specified number of minutes
    await asyncio.sleep(minutes * 60)


@bot.command()
async def playaudio(ctx, youtube_url): # play youtube video audio
    if ctx.author.voice and ctx.author.voice.channel:
        voice_channel = ctx.author.voice.channel
        voice_client = await voice_channel.connect()
        await ctx.send(f'Joined {voice_channel}')

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            url2 = info['formats'][0]['url']

        voice_client.play(discord.FFmpegPCMAudio(url2))
        while voice_client.is_playing():
            await asyncio.sleep(1)
        await voice_client.disconnect()
    else:
        await ctx.send("You need to be in a voice channel to use this command.")


token = os.getenv('DISCORD_TOKEN')
bot.run(token)

