import json
import random
import re
from pathlib import Path

import openai
from discord import Message, File
from discord.ext import commands
from discord.ext.commands import Cog

async def setup(bot):
    tts = Tts(bot)


class Tts(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        print("TTS cog init")
        self.modifier_methods = [
                    self.parse_voice
            ]

    # on message
    @Cog.listener("on_message")
    async def on_message(self, message):
        if message.author.bot:
            return

        if message.content.startswith("!*"):
            self.bot.logger.info("TTS command detected")
            file = await self.generate_ai_audio(message.content[1:])
            await message.channel.send(file=file)


    async def cog_load(self):
        print("GPT Chat cog loaded")

    def parse_voice(self, message_content, modifiers):
        match = re.search(r'\{([a-zA-Z]+)\}(.*)', message_content)
        if match:
            # Extract the voice and set it as the voice in modifiers
            voice = match.group(1)
            if voice in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
                modifiers['voice'] = voice

            # Update message_content to include only the part to the right of the bracketed section
            message_content = match.group(2).strip()

        return message_content, modifiers

    def parse_modifiers(self, message_content):
        modifiers = {
            "voice": None
        }

        for i in range(20):
            original_value = modifiers.copy()
            for modifier_method in self.modifier_methods:
                message_content, modifiers = modifier_method(message_content=message_content, modifiers=modifiers)
            if modifiers == original_value:
                break

        return message_content, modifiers

    async def generate_ai_audio(self, input_text):
        # voice = random.choice(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
        input_text, modifiers = self.parse_modifiers(input_text)
        voice = modifiers['voice']
        if not voice:
            voice = random.choice(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
        speech_file_path = Path(__file__).parent / f"audio/speech-{voice}.mp3"
        openai.api_key = self.bot.config['OPENAI_API_KEY']

        response = openai.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=input_text
        )

        response.stream_to_file(speech_file_path)

        with open(speech_file_path, "wb") as f:
            f.write(response.content)

        # To return as a Discord file object
        discord_file = File(speech_file_path)
        return discord_file

