import nextcord
import openai
import tiktoken
from beatrice.util.slash_compat import Cog
from nextcord.ext import commands
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from beatrice.main import DiscordBot


class ChatGPT(Cog):
    def __init__(self, bot: "DiscordBot"):
        self.bot = bot
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        openai.api_key = bot.config["chat_gpt"]["key"]
        self.token_budget = int(bot.config["chat_gpt"].get("token_budget", 700))

    async def __async_init__(self):
        openai.aiosession.set(self.bot.session)

    @commands.Cog.listener("on_message")
    async def on_message(self, message: nextcord.Message, *args):
        guild = message.guild
        reference = message.reference
        is_reply = False
        if reference:
            reference_guild = self.bot.get_guild(reference.guild_id)
            reference_channel = reference_guild.get_channel(reference.channel_id)
            reference_message = await reference_channel.fetch_message(reference.message_id)
            is_reply = reference_message.author == guild.me
        if (guild.me in message.mentions and message.author != guild.me) or is_reply:
            await self.reply_to(message)

    async def reply_to(self, message: nextcord.Message):
        async with message.channel.typing():
            from .prompts import prompt
            budget = self.token_budget
            budget -= len(self.enc.encode(prompt))
            log = []
            raw_messages = [message]
            async for before_message in message.channel.history(limit=10, before=message):
                raw_messages.append(before_message)
            for message in raw_messages:
                if budget < 0:
                    break
                if message.author == self.bot.user:
                    entry = {
                        "role": "assistant",
                        "content": message.content
                    }
                else:
                    entry = {
                        "role": "user",
                        "content": f"{message.author.name}: {await self.sanitize_message(message)}"
                    }
                budget -= len(self.enc.encode(entry["content"]))
                log.insert(0, entry)

            log.insert(0, {"role": "system", "content": prompt})
            completion = await openai.ChatCompletion.acreate(model="gpt-3.5-turbo", messages=log)
            result = completion.choices[0].message.content
            if result.startswith("Beatrice:"):
                result = result[len("Beatrice:"):].lstrip()
            await message.channel.send(result)

    async def sanitize_message(self, message: nextcord.Message):
        text = message.content
        for mention in message.mentions:
            text = text.replace(f"<@{mention.id}>", mention.name)
        return text

