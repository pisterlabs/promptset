from __future__ import annotations

import asyncio
import importlib
import typing
from typing import TYPE_CHECKING

import discord
from discord import app_commands
from discord.app_commands import locale_str as _T
from discord.ext import commands

import config
from core import openai as core_openai
from core.context import Context

if TYPE_CHECKING:
    from core.bot import Bot


class StudyNotes(commands.Cog):
    """
    Chat with an AI
    """

    def __init__(self, bot: Bot):
        self.openai = None
        self.bot: Bot = bot

    async def cog_load(self):
        importlib.reload(core_openai)
        from core.openai import OpenAI

        self.openai: OpenAI = OpenAI(config.OPENAI_KEY)
        self.bot.openai = self.openai

    async def cog_unload(self):
        del self.openai

    STUDY_NOTES_MAX_CONCURRENCY = commands.MaxConcurrency(1, per=commands.BucketType.member, wait=False)

    @commands.command("study-notes")
    @commands.cooldown(1, 5, commands.BucketType.member)
    async def study_notes(
        self,
        ctx: Context,
        amount: typing.Optional[commands.Range[int, 1, 10]] = 5,
        *,
        topic: str,
    ):
        """
        Generate study notes about a certain topic
        """

        await self.STUDY_NOTES_MAX_CONCURRENCY.acquire(ctx.message)

        if len(topic) > 500:
            return await ctx.send("Topic must be less than 500 characters.")

        try:
            try:
                notes = await self.openai.study_notes(topic, user=ctx.author.id, amount=amount)
            except Exception as e:
                self.bot.dispatch("command_error", ctx, e, force=True, send_msg=False)
                return await ctx.send(f"Something went wrong, try again later.")

            embed = discord.Embed(color=self.bot.color)
            embed.set_author(name="Study Notes:", icon_url=ctx.author.display_avatar.url)
            embed.title = f"Study Notes about {topic}:"
            embed.description = notes

            return await ctx.send(embed=embed)
        finally:
            await self.STUDY_NOTES_MAX_CONCURRENCY.release(ctx.message)

    @app_commands.command(name=_T("study-notes"))
    @app_commands.checks.cooldown(1, 5, key=lambda i: (i.guild_id, i.user.id))
    @app_commands.describe(
        topic=_T("The text to be checked for grammar."),
        amount=_T("The amount of study notes to be generated. Minimal 1, maximum 10."),
    )
    async def study_notes_slash(
        self,
        interaction: discord.Interaction,
        topic: str,
        amount: app_commands.Range[int, 1, 10] = 5,
    ):
        """
        Generate study notes about a certain topic
        """

        # As of right now, app_commands does not support max_concurrency, so we need to handle it ourselves in the
        # callback.
        ctx = await Context.from_interaction(interaction)
        await self.STUDY_NOTES_MAX_CONCURRENCY.acquire(ctx.message)

        if len(topic) > 500:
            return await interaction.followup.send("Topic must be less than 500 characters.", ephemeral=True)

        await interaction.response.defer()

        try:
            try:
                notes = await self.openai.study_notes(topic, user=interaction.user.id, amount=amount)
            except Exception as e:
                self.bot.tree.on_error(interaction, (e, False))
                return await interaction.followup.send(f"Something went wrong, try again later.", ephemeral=True)

            embed = discord.Embed(color=self.bot.color)
            embed.set_author(name="Study Notes:", icon_url=ctx.author.display_avatar.url)
            embed.title = f"Study Notes about {topic}:"
            embed.description = notes

            return await interaction.followup.send(embed=embed)
        finally:
            await self.STUDY_NOTES_MAX_CONCURRENCY.release(ctx.message)


async def setup(bot):
    await bot.add_cog(StudyNotes(bot))
