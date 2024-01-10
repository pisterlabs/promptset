import discord
from redbot.core import commands, Config
from openai import AsyncOpenAI
import asyncio

class DeckardCain(commands.Cog):
    """Deckard Cain as ChatGPT
    Make sure to create an API Key on [OpenAI Platform](https://platform.openai.com/)
    You will need to configure a billing method and usage limits."""

    __version__ = "1.0.5"

    def __init__(self, bot):
        self.bot = bot
        self.config = Config.get_conf(self, identifier=1928374650)
        self.config.register_guild(api_key=None, allowed_channel=None)

    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setcainapikey(self, ctx, api_key: str):
        """Sets the API Key for OpenAI ChatGPT"""

        if not ctx.channel.permissions_for(ctx.guild.me).manage_messages:
            await ctx.send("I do not have permissions to delete messages in this channel.")
            return

        await self.config.guild(ctx.guild).api_key.set(api_key)
        confirmation_message = await ctx.send("API key has been set successfully. This message will be deleted shortly.")
        await ctx.message.delete()

        await asyncio.sleep(5)
        await confirmation_message.delete()


    @commands.command()
    @commands.guild_only()
    @commands.has_permissions(administrator=True)
    async def setcainchannel(self, ctx, channel: discord.TextChannel = None):
        """Restricts `askcain` to a specified channel"""
        if channel is None:
            await self.config.guild(ctx.guild).allowed_channel.clear()
            await ctx.send("The channel restriction for `Deckard Cain` has been removed.")
        else:
            await self.config.guild(ctx.guild).allowed_channel.set(channel.id)
            await ctx.send(f"The channel '{channel.name}' has been set as the allowed channel for the `askcain` command.")

    @commands.command()
    @commands.guild_only()
    async def askcain(self, ctx, *, question):
        """Chat with Deckard Cain (ChatGPT)"""
        allowed_channel_id = await self.config.guild(ctx.guild).allowed_channel()

        if allowed_channel_id is None or ctx.channel.id == allowed_channel_id:
            api_key = await self.config.guild(ctx.guild).api_key()

            if api_key:
                response = await self.generate_response(question, api_key)
                await ctx.send(response)
            else:
                await ctx.send("API key not set! Use the command `setcainapikey`.")
        else:
            allowed_channel = self.bot.get_channel(allowed_channel_id)
            await ctx.send(f"The `askcain` command can only be used in {allowed_channel.mention}.")


    async def generate_response(self, question, api_key):
        client = AsyncOpenAI(api_key=api_key)

        prompt = (f"As Deckard Cain, the last of the Horadrim and a scholar in Sanctuary, you offer wisdom about the Diablo universe. "
                  "Your answers reflect deep knowledge of arcane lore and the eternal conflict between Heaven and Hell. "
                  "\nUser: " + question + " ")

        try:
            response = await client.completions.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=476,
                temperature=0.5
            )
            
            response_content = response.choices[0].text.strip()
            return "\n" + response_content
        except Exception as e:
            return f"An error occurred: {str(e)}"