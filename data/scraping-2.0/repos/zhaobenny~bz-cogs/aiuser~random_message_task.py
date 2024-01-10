import datetime
import logging
import random

import discord
from discord.ext import tasks

from aiuser.abc import MixinMeta
from aiuser.common.constants import DEFAULT_PROMPT
from aiuser.common.utilities import format_variables
from aiuser.messages_list.messages import create_messages_list
from aiuser.response.chat.openai import OpenAI_API_Generator
from aiuser.response.chat.response import ChatResponse

logger = logging.getLogger("red.bz_cogs.aiuser")


class RandomMessageTask(MixinMeta):
    @tasks.loop(minutes=33)
    async def random_message_trigger(self):

        if not self.openai_client:
            return
        if not self.bot.is_ready():
            return

        for guild_id, channels in self.channels_whitelist.items():
            try:
                last, ctx = await self.get_discord_context(guild_id, channels)
            except:
                continue

            guild = last.guild
            channel = last.channel

            if not await self.check_if_valid_for_random_message(guild, last):
                return


            topics = await self.config.guild(guild).random_messages_prompts() or None
            if not topics:
                return logger.warning(
                    f"No random message topics were found in {guild.name}, skipping")

            prompt = await self.config.channel(channel).custom_text_prompt() or await self.config.guild(guild).custom_text_prompt() or DEFAULT_PROMPT
            message_list = await create_messages_list(self, ctx, prompt=prompt)
            topic = format_variables(
                ctx, topics[random.randint(0, len(topics) - 1)])
            logger.debug(
                f"Sending random message to #{channel.name} at {guild.name}")
            await message_list.add_system(f"Using the persona above, follow these instructions: {topic}", index=len(message_list) + 1)
            chat = OpenAI_API_Generator(self, ctx, message_list)
            response = ChatResponse(ctx, self.config, chat)
            return await response.send(standalone=True)

    async def get_discord_context(self, guild_id: int, channels: list):
        guild = self.bot.get_guild(guild_id)

        if not channels:
            raise ValueError(f"Channels are empty in guild {guild.name}")

        channel = guild.get_channel(
            channels[random.randint(0, len(channels) - 1)])

        if not channel:
            raise ValueError(f"Channel not found in guild {guild.name}")

        last_message = await channel.fetch_message(channel.last_message_id)
        ctx = await self.bot.get_context(last_message)

        return last_message, ctx

    async def check_if_valid_for_random_message(self, guild: discord.Guild, last: discord.Message):
        if await self.bot.cog_disabled_in_guild(self, guild):
            return False
        if not (await self.bot.ignored_channel_or_guild(last)):
            return False
        if not await self.config.guild(guild).random_messages_enabled():
            return False
        if random.random() > await self.config.guild(guild).random_messages_percent():
            return False

        if last.author.id == guild.me.id:
            # skip spamming channel with random event messages
            return False

        last_created = last.created_at.replace(
            tzinfo=datetime.timezone.utc)

        if (abs((datetime.datetime.now(datetime.timezone.utc) - last_created).total_seconds())) < 3600:
            # only sent to channels with 1 hour since last message
            return False

        return True
