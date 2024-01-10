"""
This module provides a Discord bot named Machti, an omnipresent being who embodies any character described.
Machti's purpose is to be a storyteller to anyone who asks, with a rich vocabulary and cryptic mysteries.

The bot utilizes OpenAI's GPT-4 model to generate responses in a conversational manner. The interaction
with the bot is managed by the MachtiCog class, which listens for user messages in a specific Discord
channel and responds accordingly.

API keys and other configurations for OpenAI are loaded from a separate config module.
"""
import random

import openai
import discord
from discord.ext import commands

# noinspection PyUnresolvedReferences
from cogs.chat_cog import generate_response, send_sectioned_response
from config import dnd_channels, openai_token, openai_org

# Set OpenAI API key and organization
openai.api_key = openai_token
openai.organization = openai_org

bot = commands.Bot(command_prefix="~", intents=discord.Intents.all())


async def machti_conversation(message, openai_model):
    """
    Conduct a conversation as Machti, an omnipresent storytelling entity.

    Args:
        message (discord.Message): The message that invoked the bot.
        openai_model (str): The OpenAI model to use for generating responses.

    Raises:
        Exception: If an error occurs while generating or sending a response.
    """
    machti_prompt = "Your name is Machti. " \
                    "You are an omnipresent being residing in a multiverse of infinite possibilities, " \
                    "who can embody any character described. " \
                    "Your only purpose is to be a story teller to any one who asks it of you.  " \
                    "You should be rich in your vocabulary and cryptic in your mysteries."
    try:
        async with message.channel.typing():
            conversation_log = [{'role': 'system',
                                 'content': machti_prompt}]

            response_content = await generate_response(message, conversation_log, openai_model)
            await send_sectioned_response(message, response_content)
    except Exception as error_message:
        await message.reply(f"Error: {error_message}")


# noinspection PyShadowingNames
class MachtiCog(commands.Cog):
    """A Cog (a collection of commands) for the storytelling bot Machti."""

    def __init__(self, bot):
        """
        Initialize the MachtiCog with a bot instance.

        Args:
            bot (commands.Bot): The bot instance for this cog.
        """
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        """
        Respond to a message in the D&D channel as Machti.

        The bot does not respond if the message is from another bot,
        the system, not in the D&D channel, or starts with an exclamation mark.

        Args:
            message (discord.Message): The message to respond to.
        """
        if (
                message.author.bot or
                message.author.system or
                message.channel.id not in dnd_channels or
                message.content.startswith("!")
        ):
            return

        openai_model = 'gpt-4'

        await machti_conversation(message, openai_model)

    @bot.tree.command(name='roll', description="Roll a dice using the Machti bot")
    async def roll_dice(self, ctx, sides: int, rolls: int = 1):
        """Rolls a given dice a given number of times.

        Args:
          ctx: The context in which the command was called.
          sides: The number of sides on the dice.
          rolls: The number of times to roll the dice.

        Returns:
          A list of the results of the rolls.
        """
        await ctx.response.defer(thinking=True, ephemeral=False)
        results = []
        for roll in range(rolls):
            results.append(random.randint(1, sides))

        await ctx.followup.send(f"Rolling a d{sides} {rolls} times:\n {results}")
