import os
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv
from datetime import datetime
import openai

# Load the .env file to get the Discord token
load_dotenv()
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
channel_id = os.getenv('CHANNEL_ID')

# Initialize the bot
bot = commands.Bot(command_prefix='!')

# Define the path to the results folder
RESULTS_DIR = './results'

# Define the system prompt
system_prompt = (
    "You are a Bitcoin specialist, investment manager, and financial analyzer. "
    "You provide professional, insightful, and reasoning-backed analysis of the market "
    "and trading opportunities."
)

# Function to run the scraping scripts
def run_scraping_scripts():
    subprocess.run(['python', 'price_scrape.py'])
    subprocess.run(['python', 'news_scrape.py'])
    subprocess.run(['python', 'blockexplorer_scrape.py'])

# Function to get the latest file with a specific prefix
def get_latest_file(prefix):
    files = [f for f in os.listdir(RESULTS_DIR) if re.match(rf'{prefix}_\d+', f)]
    latest_file = max(files, key=lambda x: datetime.strptime(x[len(prefix) + 1:x.find('.txt')], '%Y%m%d_%H%M%S'))
    return os.path.join(RESULTS_DIR, latest_file)

# Function to read and title the file contents
def read_and_title_file(filepath, title):
    with open(filepath, 'r') as file:
        content = file.read()
    return f"{title}:\n{content}\n"

# Function to run the analysis (similar to the one in main.py but adapted for Discord)
async def run_analysis_discord():
    # Run the scraping scripts
    run_scraping_scripts()
    
    # Get the latest files
    price_file = get_latest_file('price_data')
    news_file = get_latest_file('news_analysis')
    bitcoin_file = get_latest_file('bitcoin_data')
    
    # Read and concatenate the file contents with titles
    concatenated_data = (
        read_and_title_file(price_file, 'PRICE DATA') +
        read_and_title_file(news_file, 'NEWS ANALYSIS') +
        read_and_title_file(bitcoin_file, 'BLOCK EXPLORER DATA')
    )
    
    # Define the analysis prompt
    analysis_prompt = "Please analyze the following Bitcoin data:\n" + concatenated_data
    
    # Create the chat completion
    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt}
        ],
        temperature=0.3,
        max_tokens=4000
    )
    return str(response)

# Task that schedules the daily message
@tasks.loop(hours=24)
async def scheduled_analysis():
    # Wait until the bot is ready and initialized
    await bot.wait_until_ready()
    
    # Get the current time
    now = datetime.now()
    
    # Check if the current time is 8 AM
    if now.hour == 8 and now.minute == 0:
        # Run the analysis
        analysis_response = await run_analysis_discord()
        
        # Find the channel where you want to post the analysis
        channel = bot.get_channel(channel_id)
        
        # Send the analysis response to the channel
        await channel.send(analysis_response)

# Start the loop when the bot starts
@scheduled_analysis.before_loop
async def before_scheduled_analysis():
    hour = 8
    minute = 0
    await bot.wait_until_ready()
    now = datetime.now()
    future = datetime(now.year, now.month, now.day, hour, minute)
    if now.hour >= hour and now.minute > minute:
        future += timedelta(days=1)
    await discord.utils.sleep_until(future)

# Command to manually trigger the analysis (optional)
@bot.command(name='analyze', help='Manually trigger the Bitcoin analysis')
async def analyze(ctx):
    analysis_response = await run_analysis_discord()
    await ctx.send(analysis_response)

# Event listener for when the bot has connected to Discord
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')
    scheduled_analysis.start()  # Start the scheduled task

# Run the bot with the token from the .env file
bot.run(DISCORD_TOKEN)