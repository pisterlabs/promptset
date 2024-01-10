from utils.log_config import setup_logging, logging
# from datetime import datetime, timedelta
import datetime
import openai
import asyncio
from discord.ext import tasks, commands

#Environment Variables
import os
from dotenv import load_dotenv
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

#Discord Channels
GUST_CHANNEL = 802461884091465748
ABBY_CHAT = 1103490012500201632
START_TIME = 5

class Motd(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

        #GPT Call
    async def generate_message(self):
        logger.info("[👋] Generating Message (OpenAI)")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        channel = self.bot.get_channel(ABBY_CHAT)
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system","content":  "You are a creative, thoughtful, and inspiring bunny."},
                    {"role": "user","content":  "I need your assistance, GPT! 🐇✨"},
                    {"role": "assistant", "content": "Of course, dear Abby, bunny assistant! How can I help you today? 🌟"},
                    {"role": "user", "content": "I'm looking for an inspiring and uplifting message for our creators in the Breeze Club Discord server. Can you generate a \"Message of the Day\" that will fill their hearts with joy and motivation? 🎨🌈"},
                    {"role": "assistant", "content": "Absolutely! Let me channel my creative powers and bring forth a delightful message (formatted for Discord) for your creators in the server to cherish - about 200 characters! *hops into creative mode* 🎩✨"},
                    {"role": "assistant", "content": "Message of the Day:"},
                ],
                max_tokens=300,
                temperature=1.5,
            )
            status_code = response["choices"][0]["finish_reason"]
            assert status_code == "stop", f"The status code was {status_code}."
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"[👋] Error generating message: {e}")
            await channel.send(f"Error generating MOTD message: {e}")
            return None
        
    @commands.Cog.listener()
    async def on_ready(self):
        if not self.motd_task.is_running():
            self.motd_task.start()

    @tasks.loop(hours=24)
    
    async def motd_task(self):
        logger.info("[👋] Sending MOTD")
        channel = self.bot.get_channel(GUST_CHANNEL)
        message_content = await self.generate_message()
        motd_message = f"**Message of the Day**:\n{message_content}"
        message = await channel.send(motd_message)
        await message.add_reaction('<a:z8_leafheart_excited:806057904431693824>')
        logger.info("[👋] MOTD Sent")


    @motd_task.before_loop
    async def before_motd_task(self):
        await self.bot.wait_until_ready()
        now = datetime.datetime.now()
        if now.hour < START_TIME:  # if it's before START_TIME
            wait_time = START_TIME - now.hour
        else:  # if it's past START_TIME
            wait_time = 24 - (now.hour - START_TIME)
        logger.info(
            f"[👋 ] MOTD initialized. Next message in {wait_time} hours.")
        await asyncio.sleep(wait_time * 3600) # Sleep until 5AM


async def setup(bot):
    await bot.add_cog(Motd(bot))