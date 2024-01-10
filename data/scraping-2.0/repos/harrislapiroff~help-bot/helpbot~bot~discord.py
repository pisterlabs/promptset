import discord
import openai
import logging
from typing import Iterable, Any

from helpbot.bot.openai import OpenAIReactBot


__all__ = ['DiscordBot']

logging.basicConfig(level=logging.INFO)


# Function to split messages that are above the Discord's message limit
def split_message(text: str):
    chunks = []
    while len(text) > 2000:
        index = text.rfind(". ", 0, 2000)
        if index == -1:
            index = 2000
        newline_index = text.rfind("\n", 0, index)
        if newline_index != -1:
            index = newline_index
        chunks.append(text[:index+1])
        text = text[index+1:]
    chunks.append(text)
    return chunks



class DiscordBot(OpenAIReactBot):
    def __init__(self, config: dict):
        super().__init__(config)

        assert 'discord_token' in self.config, 'Token not found in config'

        if 'user_allowlist' not in self.config:
            self.config['user_allowlist'] = []

        self.client = discord.Client(intents=discord.Intents.default())

        # Register a message handler
        @self.client.event
        async def on_message(message) -> None:
            if not self._response_allowed(message):
                return
            await self.get_response(message.content, extra_context=message)

    def _response_allowed(self, message: discord.Message) -> bool:
        logging.info(f'Saw message from {message.author} in {message.channel} with content: {message.content}')

        # If the message is from the bot itself, ignore
        if message.author == self.client.user:
            return False
        
        # If the message is from a user in the allowlist, allow
        if isinstance(message.channel, discord.channel.DMChannel) and str(message.author) in self.config['user_allowlist']:
            return True
        
        # If the message is in a channel and the bot is mentioned, allow
        if not isinstance(message.channel, discord.channel.DMChannel) and self.client.user.mentioned_in(message):
            return True
        
        # Otherwise, ignore
        return False
    
    def get_context(self, extra_context: Any):
        context = super().get_context(extra_context)
        return {**context, 'channel': extra_context.channel}


    def get_system_prompt(self, extra_context: Any):
        context = self.get_context(extra_context)
        return super().get_system_prompt(context) + (
            f'You are talking in the {context["channel"]} channel on Discord\n'
        )


    async def on_message(self, message: discord.Message) -> None:
        async with message.channel.typing():
            response = await self.get_response(message.content)
            await self.send_message(response, self.get_context(message))

    async def send_message(self, message, context: dict) -> None:
        if message != '':
            if len(message) > 2000:
                messages = split_message(message)
            else:
                messages = [message]

            for message in messages:
                await context['channel'].send(message)

    def run(self) -> None:
        self.client.run(self.config['discord_token'])