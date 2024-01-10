QUERY_LOG_FILE = 'query_logs.log'

import discord
from discord import app_commands
from discord.ext import commands
from utils.utils import createUrl

from openai import OpenAI

import datetime
import logging
from logging.handlers import RotatingFileHandler

# import asyncio
import aiohttp
import aiofiles

class AIgroup(app_commands.Group):
    openai_client = None
    """Manage general commands"""
    def __init__(self, bot: commands.Bot, config: dict[str, any]):
        """Initialize the commands related to the backend AI server.
        Args:
            bot (str): The bot client object.
            config(dict): Application config. (Key value pair)
        """

        super().__init__()

        # separate logger for debugging queries.
        my_handler = RotatingFileHandler(QUERY_LOG_FILE,
            mode = 'a',
            maxBytes = 5 * 1024 * 1024,
            backupCount = 2,
            encoding = None,
            delay = 0)
        my_handler.setLevel(logging.INFO)

        q_log = logging.getLogger('query')
        q_log.setLevel(logging.INFO)
        q_log.addHandler(my_handler)

        self.query_logger = q_log

        self.name = "ai"
        self.bot = bot
        self.config = config

        # self.openai_client = None
        if self.config['chat_gpt_fallback'] and not AIgroup.openai_client:
            AIgroup.openai_client = OpenAI()

    @app_commands.command(name = 'chat')
    async def hello(self, interaction: discord.Interaction, query: str):

        await interaction.response.defer(thinking = True)  # don't put this after the server call. The delay can be more than 3 secs and cause err on client side.

        async with aiohttp.ClientSession() as session:
            request_data = {
                'command_type': 'chat',
                "query": query
            }
            service_url = createUrl(self.config['model_server']['chat_server'],
                self.config['model_server']['chat_port'],
                self.config['model_server']['chat_uri'])

            try:
                async with session.post(service_url, data = request_data) as r:
                    if r.status == 200:
                        js = await r.json()
                        chat_msg = js.get('response', 'Looks like the server sent something weird. Gotta protect your fragile mind from it.')
                        self.query_logger.info(f'Query: {query}, response: {js}')  # logging for checking responses later.
                        return await interaction.followup.send(chat_msg)
                        # await channel.send(js['file'])
                    elif self.config['chat_gpt_fallback']:
                        msg = await self.get_completion(query)
                        return await interaction.followup.send(msg)
                    else:
                        return await interaction.followup.send('Uff kuch to galti hui...')
            except:
                if self.config['chat_gpt_fallback']:
                    msg = await self.get_completion(query)

                    return await interaction.followup.send(msg)
                else:
                    msg = await interaction.followup.send('Tera server to gaya!')


            # TODO: Add timeout for no reply case
            # try:
            #     await asyncio.wait(timeout=5)
            # except asyncio.TimeoutError:
            #     await interaction.followup.send('Tera server to gaya!')




    @app_commands.command()
    async def dream(self, interaction: discord.Interaction, query: str):
        """Sends a request to the model server and makes it generate an image."""

        async with aiohttp.ClientSession() as session:
            request_data = {
                'command_type': 'dream',
                "query": query
            }
            service_url = createUrl(self.config['model_server']['stable_diffusion_server'],
                self.config['model_server']['stable_diffusion_port'],
                self.config['model_server']['stable_diffusion_uri'])

            await interaction.response.defer(thinking = True)  # don't put this after the server call. The delay can be more than 3 secs and cause err on client side.

            async with session.post(service_url, data = request_data) as resp:
                if resp.status == 200:
                    ext = 'img'
                    if resp.content_type == 'image/png':
                        ext = 'png'
                    filename = './data/reply_img.' + ext

                    async with aiofiles.open(filename, mode='wb') as img_file:
                        await img_file.write(await resp.read())

                        await interaction.followup.send(file = discord.File(filename))

                #     chat_msg = js.get('message', 'Looks like the server sent something weird. Gotta protect your fragile mind from it.')
                #     self.query_logger.info(f'Query: {query}, response: {js}')
                #     return await interaction.followup.send(chat_msg)
                #     # await channel.send(js['file'])
                # else:
                #     return await interaction.followup.send('Uff kuch to galti hui...')

        # await interaction.response.send_message('This is not implemented yet')


    @app_commands.command(description = 'Execute commmand')
    async def execute(self, interaction: discord.Interaction, args: str):
        """Takes different commands. Send help to know more"""
        print(f'args {args}')
        if args == 'help':
            return await interaction.response.send_message('This is an not implemented yet')
        elif args.startswith('dream ') or args == 'dream':
            query = args[5:]
            if not query.strip():
                query = "Something gamey"

            return await self.dream(interaction, query)
        elif args.startswith('chat ') or args == 'chat':
            query = args[4:]
            if not query.strip():
                return await interaction.response.send_message('I need some command hooman!')

            return await self.hello(interaction, query)


        return await interaction.response.send_message('Unknown command use help for list of valid commands')


    @staticmethod
    async def get_completion(prompt, model="gpt-3.5-turbo"):
        """
        Use OpenAI API and get a response.

        Ignored most of the arguments for now.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await AIgroup.openai_client.chat.completions.create(model=model,
            messages=messages,
            temperature=0.5)

            return response.choices[0].message["content"]
        except openai.RateLimitError:
            return 'Rate limit error'
        except:
            return "Fallback error"