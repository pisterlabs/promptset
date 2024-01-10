import discord
from discord import Embed
from discord.ext import commands, tasks
import openai
import random
import asyncio
import datetime

import os
from dotenv import load_dotenv
from utils.log_config import setup_logging, logging
from utils.mongo_db import connect_to_mongodb

setup_logging()
logger = logging.getLogger(__name__)
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

ABBY_CHAT = 1103490012500201632
DAILY_GUST = 802461884091465748

class MorningAnnouncements(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.announcement_channel = self.bot.get_channel(ABBY_CHAT)
        self.daily_gust = self.bot.get_channel(DAILY_GUST)
        self.mongo = connect_to_mongodb()
        

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f"Logged in as {self.bot.user.name} - {self.bot.user.id}")
        logger.info(f"Discord.py API version: {discord.__version__}")
        logger.info(f"OpenAI API version: {openai.__version__}")
        logger.info(f"Ready to use!")
    
    @commands.command()
    async def add_to_morning(self, ctx):
        # Get the announcement
        announcement = ctx.message.content[16:]
        # Add the announcement to MongoDB
        self.mongo["Discord"]["Morning Announcements"].insert_one({"announcement": announcement})
        # Send a confirmation message
        await ctx.send(f"Added `{announcement}` to the morning announcements!")
        
    
    @commands.command()
    async def night(self, ctx):
        await ctx.send("Good night everyone! :heart:")
    
    @tasks.loop(hours=24)
    async def morning_announcements(self):
            """Send the morning announcements"""
            # Get the announcements
            announcements = self.get_announcements()
            # Send the announcements
            for announcement in announcements:
                await self.announcement_channel.send(announcement)
            # Get the daily gust
            daily_gust = self.get_daily_gust()
            # Send the daily gust
            await self.daily_gust.send(daily_gust)
            logger.info("[📢] Sent morning announcements")

    @morning_announcements.before_loop
    async def before_morning_announcements(self):
            self.bot.wait_until_ready()
            logger.info("[📢] Not time for morning announcements")
            # Get the current time
            current_time = datetime.now().strftime("%H:%M")
            # Wait until it is time for the morning announcements
            # Calculate the time until 8:00
            time_until_8 = datetime.strptime("08:00", "%H:%M") - datetime.strptime(current_time, "%H:%M")
            # Sleep until 8:00
            await asyncio.sleep(time_until_8.seconds)
            logger.info("[📢] Time for morning announcements")
            self.morning_announcements.start()
    
    def get_announcements(self):
        """Get the morning announcements"""
        # Get all of the morning announcements from MongoDB
        announcements = self.mongo["Discord"]["Morning Announcements"].find()
        # Convert the announcements to a list
        announcements = list(announcements)
        # Get the number of announcements
        num_announcements = len(announcements)
        
        return announcements,num_announcements
    
    def get_daily_gust(self):
        prompt = "The Daily Gust is a daily message from the Breeze Club Discord server."
         # Generate the Daily Gust using OpenAI
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stop=["\n"]
        )
        # Get the Daily Gust
        daily_gust = response.choices[0].text
        # Return the Daily Gust
        return daily_gust
    