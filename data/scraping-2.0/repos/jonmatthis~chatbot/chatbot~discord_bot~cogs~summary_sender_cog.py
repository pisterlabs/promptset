import logging
from datetime import datetime

import discord
from discord.ext import commands
from langchain.text_splitter import CharacterTextSplitter

from chatbot.mongo_database.mongo_database_manager import MongoDatabaseManager
from chatbot.system.filenames_and_paths import STUDENT_SUMMARIES_COLLECTION_NAME

logger = logging.getLogger(__name__)


class SummarySenderCog(commands.Cog):
    def __init__(self,
                 bot: discord.Bot,
                 mongo_database_manager: MongoDatabaseManager):
        self.bot = bot
        self.mongo_database_manager = mongo_database_manager

    @discord.slash_command(name='send_summary', description='Send user their stored summary')
    async def send_summary(self,
                           ctx: discord.ApplicationContext):
        student_discord_username = str(ctx.author)
        logger.info(f"Sending summary for {student_discord_username}...")
        # student_summary = self.mongo_database_manager.get_student_summary(discord_username=student_discord_username)
        student_summary_collection = self.mongo_database_manager.get_collection(STUDENT_SUMMARIES_COLLECTION_NAME)
        student_summary_entry = student_summary_collection.find_one({"discord_username": student_discord_username})
        summary_send_message = await ctx.send(
            f"Sending summary for {student_discord_username} as of {datetime.now().isoformat()}...")


        if student_summary_entry is None:
            await summary_send_message.edit(content=f"Could not find summary for user: {student_discord_username}...")
            await summary_send_message.add_reaction("❓")
            return

        student_summary = student_summary_entry["student_summary"]["summary"]
        created_at = student_summary_entry["student_summary"]["created_at"]
        number_of_threads = len(student_summary_entry["threads"])
        threads_created_at = [thread["created_at"].isoformat() for thread in student_summary_entry["threads"]]
        threads_created_at_str = '\n'.join(threads_created_at)
        student_summary = student_summary.replace("```", "")
        text_splitter = CharacterTextSplitter(chunk_size=1200, chunk_overlap=0)
        student_summary_split = text_splitter.split_text(student_summary)
        await ctx.user.send(
            f"Summary for {student_discord_username} as of {datetime.now().isoformat()} \n"
            f"This summary was created on:\n {created_at}\n"
            f"This summary was created from {number_of_threads} threads from:\n"
            f"{threads_created_at_str}\n")
        for chunk_number, student_summary in enumerate(student_summary_split):
            await ctx.user.send(
                f"Chunk {chunk_number + 1} of {len(student_summary_split)}:\n"
                f"\n ```\n {student_summary}\n```\n")

        await summary_send_message.edit(
            content=f"Successfully sent summary for {student_discord_username} as of {datetime.now().isoformat()}...")
        await summary_send_message.add_reaction("✅")
