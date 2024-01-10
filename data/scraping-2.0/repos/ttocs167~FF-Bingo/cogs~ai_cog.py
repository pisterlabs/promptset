import discord
from discord.ext import commands
import openai
from utilities.openAI_test import get_ai_response, get_ai_pun, get_modified_image
from utilities.chatgpt import get_chat_response, get_tokens
import os
import requests
import textwrap
import io
from PIL import Image


class AICog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(aliases=["ai_image", "aiim", "aimage"])
    @commands.is_owner()
    async def get_dalle_image(self, ctx: commands.Context, *, prompt):
        """Generate a DALLE image based on a prompt!"""
        async with ctx.channel.typing():
            if str(ctx.guild) in os.getenv('GUILD_WHITELIST'):
                try:
                    response = openai.Image.create(
                        prompt=prompt,
                        n=1,
                        size="512x512"
                    )
                    image_url = response['data'][0]['url']

                    img_data = requests.get(image_url).content

                    with open('dalle_temp_image.png', 'wb') as handler:
                        handler.write(img_data)

                    img_as_file = discord.File('dalle_temp_image.png')

                    await ctx.reply(file=img_as_file)

                except openai.error.InvalidRequestError:
                    await ctx.reply("**Your prompt has violated the content policy**")
            else:
                await ctx.reply("Sorry, this server is not authorised to use the AI function.")

    @commands.command(aliases=["aimod"])
    @commands.is_owner()
    async def ai_modify_image(self, ctx: commands.Context):
        """Modify an existing image using DALLE!"""
        async with ctx.channel.typing():
            if str(ctx.guild) in os.getenv('GUILD_WHITELIST'):
                if ctx.message.attachments:

                    file_bytes = await ctx.message.attachments[0].read()
                    image_url = get_modified_image(file_bytes)
                    img_data = requests.get(image_url).content
                    with open('dalle_temp_image.png', 'wb') as handler:
                        handler.write(img_data)

                    img_as_file = discord.File('dalle_temp_image.png')

                    await ctx.reply(file=img_as_file)

                else:
                    await ctx.reply("You need to attach an image for me to modify!")
            else:
                await ctx.reply("Sorry, this server is not authorised to use the AI function.")

    @commands.command()
    async def ai(self, ctx, *, new_prompt):
        """Get a real AI response from BingoBot!"""
        async with ctx.channel.typing():
            if str(ctx.guild) in os.getenv('GUILD_WHITELIST'):
                author = str(ctx.message.author).split('#')[0]
                response = get_chat_response(new_prompt)

                lines = textwrap.wrap(response,
                                      width=1500,
                                      expand_tabs=False,
                                      replace_whitespace=False,
                                      drop_whitespace=False,
                                      break_on_hyphens=False,
                                      break_long_words=False,
                                      )

                for line in lines:
                    await ctx.reply(line)

            else:
                await ctx.reply("Sorry, this server is not authorised to use the AI function.")

    @commands.command(hidden=True)
    @commands.is_owner()
    async def ai4(self, ctx, *, new_prompt):
        """Get a real AI response from BingoBot!"""
        async with ctx.channel.typing():
            if str(ctx.guild) in os.getenv('GUILD_WHITELIST'):
                author = str(ctx.message.author).split('#')[0]
                response = get_chat_response(new_prompt, model="gpt-4")

                lines = textwrap.wrap(response,
                                      width=1500,
                                      expand_tabs=False,
                                      replace_whitespace=False,
                                      drop_whitespace=False,
                                      break_on_hyphens=False,
                                      break_long_words=False,
                                      )

                for line in lines:
                    await ctx.reply(line)

            else:
                await ctx.reply("Sorry, this server is not authorised to use the AI function.")

    @commands.command()
    async def tokens(self, ctx, *, sample):
        """find out how many tokens are in your message!"""
        async with ctx.channel.typing():
            tokens = get_tokens(sample)
            await ctx.reply("The number of tokens in your message is: " + tokens)

    @commands.command(aliases=["ai_pun"])
    @commands.is_owner()
    async def aipun(self, ctx: commands.Context, *, pun_prompt):
        """Get a painfully unfunny AI generated pun or joke from BingoBot!"""
        async with ctx.channel.typing():
            if str(ctx.guild) in os.getenv('GUILD_WHITELIST'):
                response = get_ai_pun(pun_prompt)
                await ctx.reply(response)
            else:
                await ctx.reply("Sorry, this server is not authorised to use the AI function.")
