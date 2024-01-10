import discord
from discord.ext import commands
from pathlib import Path
from openai import OpenAI
import asyncio


class TTSCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.user_voice_channel = None
        self.user_id = None
        self.openai_client = OpenAI()  # Initialize with your API credentials

    @commands.slash_command(name="begin_tts", description="Start TTS in your voice channel.")
    async def begin_tts(self, ctx):
        if ctx.author.voice and ctx.author.voice.channel:
            self.user_voice_channel = ctx.author.voice.channel
            self.user_id = ctx.author.id
            await ctx.respond(f"Connected to voice channel: {self.user_voice_channel.name}. Send me a DM to start TTS.")
            self.bot.add_listener(self.on_message)
        else:
            await ctx.respond("You are not in a voice channel.")

    async def on_message(self, message):
        if message.author.id != self.user_id or not isinstance(message.channel, discord.DMChannel):
            return
        await self.perform_tts(message.content)

    async def perform_tts(self, text):
        speech_file_path = Path(__file__).parent / "speech.mp3"
        response = self.openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        response.stream_to_file(speech_file_path)

        if self.user_voice_channel:
            vc = await self.user_voice_channel.connect()
            vc.play(discord.FFmpegPCMAudio(executable="ffmpeg", source=str(speech_file_path)))
            while vc.is_playing():
                await asyncio.sleep(1)
            await vc.disconnect()


def setup(bot):
    bot.add_cog(TTSCog(bot))
