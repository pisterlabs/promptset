#    ╔╗╔┌─┐┬─┐┌─┐┬ ┬
#    ║║║├┤ ├┬┘│  └┬┘
#    ╝╚╝└─┘┴└─└─┘ ┴

# Code is licensed under CC-BY-NC-ND 4.0 unless otherwise specified.
# https://creativecommons.org/licenses/by-nc-nd/4.0/
# You CANNOT edit this file without direct permission from the author.
# You can redistribute this file without any changes.

# meta developer: @nercymods
# scope: hikka_min 1.6.2
# requires: pydub openai

import os

import openai
from hikkatl.tl.types import Message
from pydub import AudioSegment

from .. import loader, utils


@loader.tds
class WhisperMod(loader.Module):
    """Module for speech recognition"""

    strings = {
        "name": "WhisperMod",
        "audio_not_found": (
            "<b><emoji document_id=5818678700274617758>👮‍♀️</emoji>Not found to"
            " recognize.</b>"
        ),
        "recognized": (
            "<b><emoji"
            " document_id=5821302890932736039>🗣</emoji>Recognized:</b>\n{transcription}"
        ),
        "error": (
            "<b><emoji document_id=5980953710157632545>❌</emoji>Error occurred during"
            " transcription.</b>"
        ),
        "recognition": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Recognition...</b>"
        ),
        "downloading": "Downloading, wait",
        "autowhisper_enabled": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Auto-whisper enabled"
            " in this chat.</b>"
        ),
        "autowhisper_disabled": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Auto-whisper disabled"
            " in this chat.</b>"
        ),
    }

    strings_ru = {
        "audio_not_found": (
            "<b><emoji document_id=5818678700274617758>👮‍♀️</emoji>Не найдено, что"
            " распознавать.</b>"
        ),
        "recognized": (
            "<b><emoji"
            " document_id=5821302890932736039>🗣</emoji>Распознано:</b>\n{transcription}"
        ),
        "error": (
            "<b><emoji document_id=5980953710157632545>❌</emoji>Ошибка при"
            " транскрипции.</b>"
        ),
        "recognition": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Распознавание...</b>"
        ),
        "downloading": (
            "<b><emoji document_id=5310189005181036109>🐍</emoji>Скачивание,"
            " подождите...</b>"
        ),
        "autowhisper_enabled": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Автораспознавание"
            " включено в этом чате.</b>"
        ),
        "autowhisper_disabled": (
            "<b><emoji document_id=5307937750828194743>🫥</emoji>Автораспознавание"
            " отключено в этом чате.</b>"
        ),
    }

    def __init__(self):
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "api_key",
                None,
                lambda: "Api key for Whisper",
                validator=loader.validators.Hidden(),
            ),
            loader.ConfigValue(
                "temperature",
                "0.2",
                lambda: (
                    "The sampling temperature, between 0 and 1. Higher values like 0.8"
                    " will make the output more random, while lower values like 0.2"
                    " will make it more focused and deterministic. If set to 0, the"
                    " model will use log probability to automatically increase the"
                    " temperature until certain thresholds are hit."
                ),
                validator=loader.validators.String(),
            ),
            loader.ConfigValue(
                "prompt",
                None,
                lambda: (
                    "An optional text to guide the model's style or continue a previous"
                    " audio segment. The prompt should match the audio language."
                ),
                validator=loader.validators.String(),
            ),
        )

    @loader.command(ru_doc="распознать речь из голосового/видео сообщения в реплае")
    async def whisper(self, message: Message):
        """Transcribe speech from a voice/video message in reply"""
        rep = await message.get_reply_message()
        await message.delete()

        down = await rep.reply(self.strings["downloading"])
        file = await rep.download_media()
        file_extension = os.path.splitext(file)[1].lower()

        openai.api_key = self.config["api_key"]

        if file_extension == ".oga" or file_extension == ".ogg":
            await self.client.edit_message(
                message.chat_id, down.id, self.strings["recognition"]
            )
            input_file = file

            audio = AudioSegment.from_file(input_file, format="ogg")
            audio.export("output_file.mp3", format="mp3")

            audio_file = open("output_file.mp3", "rb")
            response = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                prompt=self.config["prompt"],
                temperature=self.config["temperature"],
            )
            response_dict = response.to_dict()
            transcription = response_dict["text"]
            await self.client.edit_message(
                message.chat_id,
                down.id,
                self.strings["recognized"].format(transcription=transcription),
            )
            os.remove(file)
            os.remove("output_file.mp3")

        elif (
            file_extension == ".mp3"
            or file_extension == "m4a"
            or file_extension == ".wav"
            or file_extension == ".mpeg"
            or file_extension == ".mp4"
        ):
            await self.client.edit_message(
                message.chat_id, down.id, self.strings["recognition"]
            )
            input_file = file

            audio_file = open(input_file, "rb")
            response = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                prompt=self.config["prompt"],
                temperature=self.config["temperature"],
            )
            response_dict = response.to_dict()
            transcription = response_dict["text"]
            await self.client.edit_message(
                message.chat_id,
                down.id,
                self.strings["recognized"].format(transcription=transcription),
            )
            os.remove(file)

        else:
            await utils.answer(message, self.strings["audio_not_found"])

    @loader.command(
        ru_doc=(
            "включить/выключить автораспознавание голосовых и видео сообщений в чате"
            " где введена команда"
        )
    )
    async def autowhspr(self, message: Message):
        """Enable/disable auto-speech recognition for voice and video messages"""
        chat_id = str(message.chat_id)
        current_state = self.get("autowhspr", {})
        enabled = current_state.get(chat_id, False)

        if enabled:
            current_state.pop(chat_id, None)
            status_message = self.strings["autowhisper_disabled"]
        else:
            current_state[chat_id] = True
            status_message = self.strings["autowhisper_enabled"]

        self.set("autowhspr", current_state)
        await utils.answer(message, status_message)

    @loader.watcher(only_media=True)
    async def autowhisper_watcher(self, message: Message):
        """Watcher to automatically transcribe voice and video messages when auto-speech recognition is enabled"""
        chat_id = str(message.chat_id)
        current_state = self.get("autowhspr", {})

        if current_state.get(chat_id, False):
            if message.voice or message.video:
                if not message.gif and not message.sticker and not message.photo:
                    rep = message
                    await self.whisperwatch(rep)

    async def whisperwatch(self, rep: Message):
        """Transcribe speech from a voice/video message in reply"""
        down = await rep.reply(self.strings["downloading"])
        file = await rep.download_media()
        file_extension = os.path.splitext(file)[1].lower()

        openai.api_key = self.config["api_key"]

        if file_extension == ".oga" or file_extension == ".ogg":
            await self.client.edit_message(
                rep.chat_id, down.id, self.strings["recognition"]
            )
            input_file = file

            audio = AudioSegment.from_file(input_file, format="ogg")
            audio.export("output_file.mp3", format="mp3")

            audio_file = open("output_file.mp3", "rb")
            response = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                prompt=self.config["prompt"],
                temperature=self.config["temperature"],
            )
            response_dict = response.to_dict()
            transcription = response_dict["text"]
            await self.client.edit_message(
                rep.chat_id,
                down.id,
                self.strings["recognized"].format(transcription=transcription),
            )
            os.remove(file)
            os.remove("output_file.mp3")

        elif (
            file_extension == ".mp3"
            or file_extension == "m4a"
            or file_extension == ".wav"
            or file_extension == ".mpeg"
            or file_extension == ".mp4"
        ):
            await self.client.edit_message(
                rep.chat_id, down.id, self.strings["recognition"]
            )
            input_file = file

            audio_file = open(input_file, "rb")
            response = openai.Audio.transcribe(
                "whisper-1",
                audio_file,
                prompt=self.config["prompt"],
                temperature=self.config["temperature"],
            )
            response_dict = response.to_dict()
            transcription = response_dict["text"]
            await self.client.edit_message(
                rep.chat_id,
                down.id,
                self.strings["recognized"].format(transcription=transcription),
            )
            os.remove(file)

        else:
            await utils.answer(rep, self.strings["audio_not_found"])
