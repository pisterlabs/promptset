import json
import asyncio
import openai
import discord
from discord import app_commands

import logger

with open('./config.json') as data:
    config = json.load(data)

path = config['LOG_PATH']
openai.api_key = config['OPENAI_API_KEY']


class aclient(discord.Client):

    def __init__(self):
        super().__init__(intents=discord.Intents.all())
        self.synced = False

    async def on_ready(self):

        await self.wait_until_ready()

        await self.change_presence(activity=discord.Activity(
            type=discord.ActivityType.listening,
            name='your prompts!'
        ))

        if not self.synced:
            await tree.sync()
            self.synced = True

        print('Logged in as:')
        print(f'{self.user.name}, {self.user.id}')
        print('Created by Joeyy#4628. The most up-to-date code can be found on github: https://github.com/Joeya1ds/Discord-GPT3-Bot')
        print('--------------------------------------------------------------------------------------------------------------------')


client = aclient()
tree = app_commands.CommandTree(client)


@tree.command(name='ask', description='Ask the AI bot a question!')
async def ask(interaction: discord.Interaction, prompt: str):

    user_id = interaction.user.id
    await interaction.response.defer()

    # Moderation API flagging
    moderate = openai.Moderation.create(input=prompt)
    flagged = moderate['results'][0]['flagged']
    
    # Functions for generating and sending bot responses to chat messages
    if flagged:

        logger.create_log_file(path, user_id)

        logger.append_log(path, user_id, prompt)
        logger.append_warning_log(path, user_id)
        logger.append_newline_log(path, user_id)

        await asyncio.sleep(3)
        await interaction.followup.send('I cannot respond to what you have said, it has been flagged by the Moderation API.')
        print(f'User with ID: {user_id} has had a prompt flagged by the Moderation API. Consider checking logs.')
        
        return

    openairesponse = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=200,
        temperature=0.8,
        top_p=0.8,
    )

    logger.create_log_file(path, user_id)

    logger.append_log(path, user_id, prompt)
    logger.append_token_log(path, user_id, openairesponse["usage"]["total_tokens"])
    logger.append_newline_log(path, user_id)

    await asyncio.sleep(3)
    await interaction.followup.send(openairesponse['choices'][0]['text'])


client.run(config['DISCORD_BOT_TOKEN'])
