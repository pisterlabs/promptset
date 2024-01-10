import discord
from discord import default_permissions
from discord.ext import commands
import os
import openai
import requests


class Moderation(discord.Cog):
    def __init__(self, bot: discord.Bot) -> None:
        super().__init__()
        self.bot = bot

    """
    @discord.slash_command(
        name="ban", description="Ban a user from using the bot"
    )
    @commands.is_owner()
    async def ban(self, ctx: discord.ApplicationContext, user: discord.User):
        pass
    """

    @discord.slash_command(
        name="moderation", description="Enable or disable AI moderation & set the rules"
    )
    @discord.option(
        name="enable",
        description="Enable or disable AI moderation",
        reqired=True,
    )
    @discord.option(
        name="log_channel",
        description="The channel where the moderation logs will be sent",
        required=True,
    )
    @discord.option(
        name="moderator_role", description="The role of the moderators", required=True
    )
    # the types of toxicity are     'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'IDENTITY_ATTACK': {}, 'INSULT': {}, 'PROFANITY': {}, 'THREAT': {}, 'SEXUALLY_EXPLICIT': {}, 'FLIRTATION': {}, 'OBSCENE': {}, 'SPAM': {}},
    @discord.option(
        name="toxicity", description="The toxicity threshold", required=False
    )
    @discord.option(
        name="severe_toxicity",
        description="The severe toxicity threshold",
        required=False,
    )
    @discord.option(
        name="identity_attack",
        description="The identity attack threshold",
        required=False,
    )
    @discord.option(name="insult", description="The insult threshold", required=False)
    @discord.option(
        name="profanity", description="The profanity threshold", required=False
    )
    @discord.option(name="threat", description="The threat threshold", required=False)
    @discord.option(
        name="sexually_explicit",
        description="The sexually explicit threshold",
        required=False,
    )
    @discord.option(
        name="flirtation", description="The flirtation threshold", required=False
    )
    @discord.option(name="obscene", description="The obscene threshold", required=False)
    @discord.option(name="spam", description="The spam threshold", required=False)
    # we set the default permissions to the administrator permission, so only the server administrators can use this command
    @default_permissions(administrator=True)
    async def moderation(
        self,
        ctx: discord.ApplicationContext,
        enable: bool,
        log_channel: discord.TextChannel,
        moderator_role: discord.Role,
        toxicity: float = None,
        severe_toxicity: float = None,
        identity_attack: float = None,
        insult: float = None,
        profanity: float = None,
        threat: float = None,
        sexually_explicit: float = None,
        flirtation: float = None,
        obscene: float = None,
        spam: float = None,
    ):
        # local import, because we don't want to import the toxicity function if the moderation is disabled
        # import toxicity as tox  # this is a file called toxicity.py, which contains the toxicity function that allows you to check if a message is toxic or not (it uses the perspective api)
        await ctx.respond(
            "Our moderation capabilities have been switched to our new 100% free and open-source AI discord moderation bot! You add it to your server here: https://discord.com/api/oauth2/authorize?client_id=1071451913024974939&permissions=1377342450896&scope=bot and you can find the source code here: https://github.com/Paillat-dev/Moderator/ \n If you need help, you can join our support server here: https://discord.gg/pB6hXtUeDv",
            ephemeral=True,
        )

    @discord.slash_command(
        name="get_toxicity", description="Get the toxicity of a message"
    )
    @discord.option(
        name="message", description="The message you want to check", required=True
    )
    @default_permissions(administrator=True)
    async def get_toxicity(self, ctx: discord.ApplicationContext, message: str):
        await ctx.respond(
            "Our moderation capabilities have been switched to our new 100% free and open-source AI discord moderation bot! You add it to your server here: https://discord.com/api/oauth2/authorize?client_id=1071451913024974939&permissions=1377342450896&scope=bot and you can find the source code here: https://discord.gg/pB6hXtUeDv . If you need help, you can join our support server here: https://discord.gg/pB6hXtUeDv",
            ephemeral=True,
        )

    @discord.slash_command(
        name="moderation_help", description="Get help with the moderation AI"
    )
    @default_permissions(administrator=True)
    async def moderation_help(self, ctx: discord.ApplicationContext):
        await ctx.respond(
            "Our moderation capabilities have been switched to our new 100% free and open-source AI discord moderation bot! You add it to your server here: https://discord.com/api/oauth2/authorize?client_id=1071451913024974939&permissions=1377342450896&scope=bot and you can find the source code here: https://github.com/Paillat-dev/Moderator/ . If you need help, you can join our support server here: https://discord.gg/pB6hXtUeDv",
            ephemeral=True,
        )
