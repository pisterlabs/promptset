from dataclasses import dataclass
import logging
from pathlib import Path
from random import random
import discord
import re
import base64
from discord.ext import commands
from marsbots.models import ChatMessage
from marsbots.language_models import complete_text
from marsbots.discord_utils import (
    get_discord_messages,
    get_nick,
    is_mentioned,
    replace_mentions_with_usernames,
    wait_for_user_reply,
)
from marsbots.language_models import OpenAIGPT3LanguageModel
from marsbots.settings_manager import LocalSettingsManager


@dataclass
class DoomerSettings:
    engine: str = "davinci"
    temperature: float = 1.0
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.2
    autoreply_probability: float = 0.01


class DoomerCog(commands.Cog):
    def __init__(self, bot: commands.bot) -> None:
        self.bot = bot
        self.settings_path = Path("../doomer_settings.json")
        self.settings_manager = LocalSettingsManager(
            self.settings_path,
            defaults=DoomerSettings(),
        )
        self.prohibited_words = self.read_prohibited_words("filtered_words.txt")

    @commands.Cog.listener("on_message")
    async def on_message(self, message: discord.Message) -> None:
        ctx = await self.bot.get_context(message)
        if (
            is_mentioned(message, self.bot.user)
            or self.bot.settings.name.lower() in message.content.lower()
            or (not message.author.bot and self.should_act(ctx))
        ):
            ctx = await self.bot.get_context(message)
            async with ctx.channel.typing():
                completion = await self.get_completion_with_chat_context(ctx, 10)
                await ctx.channel.send(completion)

    @commands.slash_command()
    async def respond(
        self,
        ctx: commands.Context,
        n_messages: discord.Option(
            int,
            description="Number of recent messages to include",
            required=False,
            default=10,
        ),
    ) -> None:
        await ctx.defer()
        completion = await self.get_completion_with_chat_context(ctx, n_messages)
        await ctx.respond(completion)

    @commands.slash_command()
    async def complete(
        self,
        ctx,
        prompt: discord.Option(
            str,
            description="Text to complete",
            required=True,
        ),
        max_tokens: discord.Option(
            int,
            description="Number of tokens to generate",
            required=False,
            default=100,
        ),
    ):
        await ctx.defer()
        language_model = self.get_language_model(ctx)
        completion = await complete_text(language_model, prompt, max_tokens=max_tokens)
        # completion_filtered = await self.filter_completion(completion)
        formatted = f"{prompt}{completion}"
        await ctx.respond(formatted)

    def should_act(self, ctx) -> bool:
        r = random()
        channel_id, guild_id = ctx.channel.id, ctx.guild.id
        autoreply_probability = self.settings_manager.get_channel_setting(
            channel_id, guild_id, "autoreply_probability"
        )
        return r < autoreply_probability

    async def get_completion_with_chat_context(self, ctx, n_messages):
        prompt = await self.format_prompt(ctx, n_messages)
        language_model = self.get_language_model(ctx)
        completion = await complete_text(
            language_model, prompt, max_tokens=300, stop=["**["]
        )
        return completion
        # completion_filtered = await self.filter_completion(self, completion)
        # return completion_filtered

    async def filter_completion(self, completion: str) -> str:
        regex = re.compile("|".join(map(re.escape, self.prohibited_words)))
        return regex.sub("####", completion)

    def get_language_model(self, ctx):
        channel_id, guild_id = ctx.channel.id, ctx.guild.id
        frequency_penalty = self.settings_manager.get_channel_setting(
            channel_id, guild_id, "frequency_penalty"
        )
        presence_penalty = self.settings_manager.get_channel_setting(
            channel_id, guild_id, "presence_penalty"
        )
        return OpenAIGPT3LanguageModel(
            engine=self.settings_manager.get_channel_setting(
                channel_id, guild_id, "engine"
            ),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )

    async def format_prompt(self, ctx, n_messages):
        last_messages = await get_discord_messages(ctx.channel, n_messages)
        last_messages = (
            last_messages[:-1]
            if last_messages[-1].type == discord.MessageType.application_command
            else last_messages
        )
        prompt = "\n".join(
            [
                str(
                    ChatMessage(
                        self.message_preprocessor(m),
                        get_nick(m.author),
                        deliniator_left=f"**[{m.created_at.strftime('%I:%M:%S %p')} ",
                    )
                ).strip()
                for m in last_messages
            ]
        )
        prompt += "\n"
        prompt += f"**[{self.bot.user.display_name}]**:"
        print(prompt)
        return prompt

    def message_preprocessor(self, message: discord.Message) -> str:
        message_content = replace_mentions_with_usernames(
            message.content, message.mentions, prefix="@"
        )
        message_content = message_content.strip()
        return message_content

    def read_prohibited_words(self, filename: str) -> list[str]:
        try:
            return (
                base64.b64decode(open(filename, "r").read())
                .decode("utf-8")
                .split("\r\n")
            )
        except OSError:
            logging.error(f"Unable to open file: {filename}")
            return [""]

    # SETTINGS

    @commands.slash_command()
    async def update_settings(
        self,
        ctx,
        setting: discord.Option(
            str,
            description="Setting name to update",
            required=True,
            choices=list(DoomerSettings.__dataclass_fields__.keys()),
        ),
        channel_name: discord.Option(
            str,
            description="Channel to update setting for",
            required=False,
        ),
    ):

        if channel_name:
            await self.handle_update_channel_settings(ctx, setting, channel_name)
        else:
            await self.handle_update_settings(ctx, setting)

    async def handle_update_settings(self, ctx, setting):
        await ctx.respond(
            f"Enter a new value for {setting}. (Currently"
            f" {self.settings_manager.get_setting(ctx.guild.id, setting)})",
        )
        resp = await wait_for_user_reply(self.bot, ctx.author.id)
        try:
            new_val = DoomerSettings.__dataclass_fields__[setting].type(
                resp.content,
            )
        except ValueError:
            await ctx.send(f"{resp.content} is not a valid value for {setting}")
            return
        self.settings_manager.update_setting(ctx.guild.id, setting, new_val)
        await ctx.send(f"Updated {setting} to {new_val}")

    async def handle_update_channel_settings(self, ctx, setting, channel_name):
        channel = discord.utils.get(ctx.guild.channels, name=channel_name)
        if not channel:
            await ctx.respond(f"No channel named {channel_name}")
            return

        await ctx.respond(
            f"Enter a new value for {setting}. (Currently"
            f" {self.settings_manager.get_channel_setting(channel.id, ctx.guild.id, setting)})",
        )
        resp = await wait_for_user_reply(self.bot, ctx.author.id)
        try:
            new_val = DoomerSettings.__dataclass_fields__[setting].type(
                resp.content,
            )
        except ValueError:
            await ctx.send(f"{resp.content} is not a valid value for {setting}")
            return
        self.settings_manager.update_channel_setting(
            channel.id,
            ctx.guild.id,
            setting,
            new_val,
        )
        await ctx.send(f"Updated {setting} to {new_val}")


def setup(bot: commands.Bot) -> None:
    bot.add_cog(DoomerCog(bot))
