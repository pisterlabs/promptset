import discord, os
from discord.ext import commands
from discord import app_commands
import FreshBot as fb
import asyncio

import openai

# Enable or Disable chatbot features
ENABLED = True

# ChatGPT parameters
ENGINE = "text-davinci-003"
TEMPERATURE = 0.9
MAX_TOKENS = 150
PRESENCE_PENALTY = 0.6

class ChatGPT(commands.Cog):
    def __init__(self, bot):
        # Load variables
        openai.api_key = os.getenv('OPENAI_API_KEY')
        self.bot = bot

    def discord_requester(self, interaction: discord.Interaction):
        """Returns the discord user who requested the song"""
        name = interaction.user.name
        discriminator = interaction.user.discriminator
        mention = interaction.user.mention
        return f'{mention}'
    
    # Send message to channel depending on if chatbot is enabled
    async def send_message(self, interaction: discord.Interaction, message):
        """ Send message to channel depending on if chatbot is enabled """
        if ENABLED:
            await interaction.followup.send(message)
        else:
            await interaction.response.send_message("Chatbot is disabled.")

    # Get response from GPT API
    async def get_response(self, message):
        """ Get response from GPT API"""
        try:
            response = openai.Completion.create(
                engine=ENGINE,
                prompt=message,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                presence_penalty=PRESENCE_PENALTY,
            )
            return response.choices[0].text
        except Exception as e:
            print("chatGPT Error: ", e)
            return "Error getting response"

    # Ask something (guild only)
    # @commands.guild_only()
    @app_commands.command(name='ask')
    async def ask(self, interaction: discord.Interaction, *, message: str):
        """ Ask the bot something """
        msg = self.discord_requester(interaction) + " asked: \n" + message
        await interaction.response.send_message(msg)
        await self.send_message(interaction, await self.get_response(message))
        
    # @commands.dm_only()
    # @commands.command(name='query')
    # async def ask(self, ctx, *, message):
    #     """ Ask the bot something """
    #     await self.send_message(ctx, await self.get_response(message))

async def setup(bot):
    await bot.add_cog(ChatGPT(bot))