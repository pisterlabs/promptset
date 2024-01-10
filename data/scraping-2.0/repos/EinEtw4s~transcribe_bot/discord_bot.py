import openai
import ffmpeg
import os
import discord
import asyncio
import textwrap
import configparser
from pathlib import Path

Path("temp/").mkdir(parents=True, exist_ok=True)

config = configparser.ConfigParser()
config.read('config.ini')

openai.api_key = config['DEFAULT']['OpenAIKey'] # your OpenAI-API Key
allowed_senders = config['DEFAULT']['AllowedDiscordSenders'].split(',') # list of allowed Discord accounts
allowed_channel_names = config['DEFAULT']['AllowedDiscordChannels'].split(',') # allowed channel names

class TranscribeClient(discord.Client):
    async def on_ready(self):
        print(f'Logged on as {self.user}!')

    async def on_message(self, message):
        # check if message should be processed and has attachment
        if str(message.author) in allowed_senders and str(message.channel) in allowed_channel_names:
            if len(message.attachments) < 1:
                return

            for attachment in message.attachments:
                # has to have audio
                if attachment.content_type.split('/')[0] not in ['audio', 'video']:
                    continue

                await attachment.save('temp/in')

                # convert to mp3
                input = ffmpeg.input('temp/in')
                out = ffmpeg.output(input,'temp/out.mp3')
                ffmpeg.run(out, overwrite_output=True)
                audio_file = open("temp/out.mp3", "rb")

                # transcribe using OAI Whisper
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                text = transcript['text']

                # respond with the transcription
                await message.channel.send(f'**{attachment.filename}:**')
                for t in textwrap.wrap(text, 2000):
                    await message.channel.send(t)
            return


async def main():
    return


if __name__=='__main__':
    token = config['DEFAULT']['DiscordToken']
    intents = discord.Intents.default()
    intents.message_content = True

    client = TranscribeClient(intents=intents)
    client.run(token)
