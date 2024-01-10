import discord
from discord.ext import commands
import openai
import logging

class OpenAiCog(commands.Cog):
    def __init__(self, client):
        self.client = client
        self.config = client.config

        openai.api_key = self.config['OPENAI_API_KEY']
    
    @commands.command()
    async def openai_image(self, ctx: commands.Context, *, arg: str):
        user = ctx.message.author
        logging.info(f'Received openai image command from user: {user.name}.')
        logging.info(f'openai image argument: {arg}.')

        try:
            response = openai.Image.create(
                prompt=arg,
                n=1,
                size="512x512"
            )
            image_url = response['data'][0]['url']

            embed = discord.Embed(title='openai_image says:', description=arg)
            embed.set_image(url=image_url)
            embed.set_author(name=user.name)

            await ctx.message.channel.send(embed=embed, delete_after=3600)
        except Exception as e:
            logging.error(e)
    
    @commands.command()
    async def openai_text(self, ctx: commands.Context, *, arg: str):
        user = ctx.message.author
        logging.info(f'Received openai text command from user: {user.name}.')
        logging.info(f'openai text argument: {arg}.')

        try:
            response = openai.Completion.create(
                model='text-davinci-003',
                prompt=arg,
                max_tokens=64
            )
            response_text = response['choices'][0]['text']
            await ctx.message.channel.send(f'openai_text says:\n{response_text}')

            logging.info(f'''openai usage:
    Prompt tokens: {response['usage']['prompt_tokens']}
    Completion tokens: {response['usage']['completion_tokens']}
    Total tokens: {response['usage']['total_tokens']}''')

        except Exception as e:
            logging.error(e)




async def setup(client: commands.Bot):
    await client.add_cog(OpenAiCog(client))