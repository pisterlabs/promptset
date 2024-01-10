import re
from tempfile import NamedTemporaryFile

import discord
import openai
from redbot.core import Config, app_commands, commands
from redbot.core.commands import Context


class LLMMixin:
    """Class to mixin because I'm too lazy to make it its own full cog"""

    async def set_settings(self):
        """Sets the settings."""
        async with self.conf.ai() as config:
            openai.api_base = config.get("url")
            openai.api_key = config.get("api_key")
            self.ai_name = config.get("name")
            self.ai_model = config.get("model")
            self.ai_system_message = config.get("system_message")

    @commands.is_owner()
    @commands.command()
    async def ai_set_settings(self, ctx, setting_name, value):
        """Set the settings"""
        config: dict
        async with self.conf.ai() as config:
            if setting_name not in config.keys():
                await ctx.reply(f"{setting_name} not in settings. Options are: {config.keys()}")
            if setting_name == "enabled":
                value = bool(value.lower() == "true")
            if setting_name == "system_message":
                if value.lower() == "none":
                    value = None
            config[setting_name] = value
        await self.set_settings()
        await ctx.message.add_reaction(self.CHECK_MARK)

    @commands.Cog.listener("on_message_without_command")
    async def local_ai_talker(self, message: discord.Message):
        """Responds to messages that start with 'Hoobot,'"""
        if self.ai_name is None:
            await self.set_settings()
        check_name = f"{self.ai_name},".lower()

        if not message.content.lower().startswith(check_name):
            return
        if message.author.bot:
            return
        ctx = await self.bot.get_context(message)

        if message.content.lower() == f"{check_name} reset":
            await ctx.message.add_reaction(self.CHECK_MARK)
            return

        messages = []
        async for msg in ctx.channel.history(limit=20):
            if message.id == msg.id:  # We want to add the actual prompt later.
                continue
            if msg.author.bot:
                if msg.content.startswith("This is an AI response from Hoobot"):
                    messages.append(
                        {
                            "role": "system",
                            "content": re.split(
                                f"^This is an AI response from {self.ai_name}:\n\n",
                                msg.content,
                            )[1],
                            "name": msg.author.name,
                        }
                    )
            else:
                if msg.content.lower() == f"{check_name} reset":
                    break
                if msg.content.lower().startswith(check_name):
                    messages.append(
                        {
                            "role": "system",
                            "content": re.split("^hoobot, ?", msg.content, flags=re.IGNORECASE)[1],
                            "name": msg.author.name,
                        }
                    )
            if len(messages) >= 5:
                break
        if self.ai_system_message:
            messages.append(
                {
                    "role": "system",
                    "content": self.ai_system_message,
                }
            )
        messages.reverse()
        messages.append(
            {
                "role": "user",
                "content": re.split("^hoobot, ?", message.content, flags=re.IGNORECASE)[1],
                "name": message.author.name,
            }
        )
        async with ctx.typing():
            chat_completion = await openai.ChatCompletion.acreate(
                model=self.ai_model,
                messages=messages,
                frequency_pentalty=1.2,
                stop="### Response:",
            )
            try:
                response = f"This is an AI response from Hoobot:\n\n{chat_completion.choices[0].message.content}"
                await ctx.reply(response, mention_author=False)
            except AttributeError:
                await ctx.reply(
                    f"There was no response from the AI. Try again, or say '{check_name} reset' to restart the conversation.",
                    mention_author=False,
                )
            except openai.APIError:
                await ctx.reply(
                    f"The AI errored! Try waiting a couple minutes, then say '{check_name} reset' and try again",
                    mention_author=False,
                )
