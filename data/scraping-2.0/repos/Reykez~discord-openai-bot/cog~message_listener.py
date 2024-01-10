import discord
from discord.ext import commands
import conversations_manager as cm
from openai_connector import map_conversation, get_chat_response
from settings import *


class MessageListenerCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user or not isinstance(message.channel, discord.TextChannel) or message.channel.category_id != channel_category_id or message.content[0] == '$':
            #await self.bot.process_commands(message)
            return

        channel_id = message.channel.id
        if channel_id not in cm.conversations:
            cm.conversations[channel_id] = cm.create_or_restore_conversation(channel_id)

        messages = map_conversation(cm.conversations[channel_id], message.content)
        response = get_chat_response(messages)

        cm.conversations[channel_id].append({
            "message": message.content,
            "response": response
        })
        cm.save_conversation(cm.conversations[channel_id], channel_id)
        await message.channel.send(response)


async def setup(bot):
    await bot.add_cog(MessageListenerCog(bot))
