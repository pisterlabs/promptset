import asyncio
import io
import logging
import subprocess
from typing import Tuple

import aiohttp
import openai
from aiohttp import ClientResponseError
from pydub.audio_segment import AudioSegment
from pydub.exceptions import CouldntDecodeError
from telegram import Update, Voice, Audio, Video, VideoNote, File
from telegram.constants import ParseMode

import os

from bobweb.bob import database, openai_api_utils, async_http
from bobweb.bob.openai_api_utils import notify_message_author_has_no_permission_to_use_api
from bobweb.bob.utils_common import object_search
from bobweb.web.bobapp.models import Chat

logger = logging.getLogger(__name__)

converter_audio_format = 'mp4'  # Default audio format that is used for converted audio file sent to openai api


def is_ffmpeg_available():
    """ Checks if ffmpeg is available in the running environment
        Calls 'ffmpeg --version' ins sub process to check if ffmpeg is available in path.
        Returns true if available """
    try:
        subprocess.check_call(['ffmpeg', '-version'])
        return True  # No error, ffmpeg is available
    except Exception:
        return False  # Error, ffmpeg not available


def notify_if_ffmpeg_not_available():
    if not ffmpeg_available:
        warning = 'NOTE! ffmpeg program not available. Command depending on video- and/or ' \
                  'audio conversion won\'t work. To enable, install ffmpeg and make it runnable' \
                  'from the terminal / command prompt.'
        logger.warning(warning)


# Checks if FFMPEG is installed in the system
ffmpeg_available = is_ffmpeg_available()

# Gives warning if ffmpeg is not available
notify_if_ffmpeg_not_available()


class TranscribingError(Exception):
    """ Any error raised while handling audio media file or transcribing it """
    def __init__(self, reason: str, additional_log_content: str = None):
        super(TranscribingError, self).__init__()
        self.reason = reason
        self.additional_log_content = additional_log_content


async def handle_voice_or_video_note_message(update: Update):
    """
    Handles any voice or video note message sent to a chat. Only processes it, if automatic transcribing is set to be
    on in the chat settings

    Transcribing: Transcribes voice to text using OpenAi's Whisper api. Requires that the user has permission
                  to use the api
    """

    chat: Chat = database.get_chat(update.effective_chat.id)
    if chat.voice_msg_to_text_enabled:
        has_permission = openai_api_utils.user_has_permission_to_use_openai_api(update.effective_user.id)
        if not has_permission:
            await notify_message_author_has_no_permission_to_use_api(update)
        else:
            await transcribe_and_send_response(update, update.effective_message.voice)


async def transcribe_and_send_response(update: Update, media_meta: Voice | Audio | Video | VideoNote):
    """
    "Controller" of media transcribing. Handles invoking transcription call,
    replying with transcription and handling error raised from the process
    """
    response = 'Median tekstittäminen ei onnistunut odottamattoman poikkeuksen johdosta.'
    try:
        transcription = await transcribe_voice(media_meta)
        cost_str = openai_api_utils.state.add_voice_transcription_cost_get_cost_str(media_meta.duration)
        response = f'"{transcription}"\n\n{cost_str}'
    except CouldntDecodeError as e:
        logger.error(e)
        response = 'Ääni-/videotiedoston alkuperäistä tiedostotyyppiä tai sen sisältämää median ' \
                   'koodekkia ei tueta, eikä sitä näin ollen voida tekstittää.'
    except TranscribingError as e:
        logger.error(f'TranscribingError: {e.additional_log_content}')
        response = f'Median tekstittäminen ei onnistunut. {e.reason or ""}'
    except Exception as e:
        logger.error(e)
    finally:
        await update.effective_message.reply_text(response, parse_mode=ParseMode.HTML)


async def transcribe_voice(media_meta: Voice | Audio | Video | VideoNote) -> str:
    """
    Downloads, converts and transcribes given Telegram audio or video object.

    NOTE! May raise Exception
    :param media_meta: media, which is transcribed
    :return: transcription of given media
    """

    # 1. Get the file metadata and file proxy from Telegram servers
    file_proxy: File = await media_meta.get_file()

    # 2. Create bytebuffer and download the actual file content to the buffer.
    #    Telegram returns voice message files in 'ogg'-format
    with io.BytesIO() as buffer:
        await file_proxy.download_to_memory(out=buffer)
        buffer.seek(0)

        # 3. Convert audio to mp4 if not yet in that format
        original_format = convert_file_extension_to_file_format(get_file_type_extension(file_proxy.file_path))
        buffer, written_bytes = convert_buffer_content_to_audio(buffer, original_format)

        max_bytes_length = 1024 ** 2 * 25  # 25 MB
        if written_bytes > max_bytes_length:
            reason = f'Äänitiedoston koko oli liian suuri.\n' \
                     f'Koko: {get_mb_str(written_bytes)} MB. Sallittu koko: {get_mb_str(max_bytes_length)} MB.'
            raise TranscribingError(reason)

        # 6. Prepare request parameters and send it to the api endpoint. Http POST-request is used
        #    instead of 'openai' module, as 'openai' module does not support sending byte buffer as is
        url = 'https://api.openai.com/v1/audio/transcriptions'
        headers = {'Authorization': 'Bearer ' + openai.api_key}

        # Create a FormData object to send files
        form_data = aiohttp.FormData()
        form_data.add_field('model', 'whisper-1')
        form_data.add_field('file', buffer, filename=f'{file_proxy.file_id}.{converter_audio_format}')

        try:
            content: dict = await async_http.post_expect_json(url, headers=headers, data=form_data)
            return object_search(content, 'text')
        except ClientResponseError as e:
            reason = f'OpenAI:n api vastasi pyyntöön statuksella {e.status}'
            additional_log = f'Openai /v1/audio/transcriptions request returned with status: ' \
                             f'{e.status}. Response text: \'{e.message}\''
            raise TranscribingError(reason, additional_log)


def convert_file_extension_to_file_format(file_extension: str) -> str:
    return (file_extension
            .replace('oga', 'ogg')
            .replace('ogv', 'ogg')
            .replace('ogx', 'ogg')
            .replace('3gp2', '3gp')
            .replace('3g2', '3gp')
            .replace('3gpp', '3gp')
            .replace('3gpp2', '3gp')
            .replace('m4a', 'aac')
            )


def convert_buffer_content_to_audio(buffer: io.BytesIO, from_format: str) -> Tuple[io.BytesIO, int]:
    """
    Return tuple of buffer and written byte count.
    More information about bydup in https://github.com/jiaaro/pydub/blob/master/API.markdown

    :param buffer: buffer that contains original audio file bytes
    :param from_format: original format
    :return: tuple (buffer, byte count)
    """
    # 1. Create AudioSegment from the byte buffer with format information
    original_version = AudioSegment.from_file(buffer, format=from_format)

    # 2. Reuse buffer and overwrite it with converted wav version to the buffer
    parameters = ['-vn']  # ffmpeg parameter -vn: no video, only audio
    original_version.export(buffer, format=converter_audio_format, parameters=parameters)

    # 3. Check file size limit after conversion. Uploaded audio file can be at most 25 mb in size.
    #    As 'AudioSegment.export()' seeks the buffer to the start we can get buffer size with (0, 2)
    #    which does not copy whole buffer to the memory
    written_bytes = buffer.seek(0, 2)
    buffer.seek(0)  # Seek buffer back to the start
    return buffer, written_bytes


def get_file_type_extension(filename: str) -> str | None:
    parts = os.path.splitext(filename)
    if parts and len(parts) > 1:
        return parts[1].replace('.', '')
    return None


def format_float_str(value: float, precision: int = 2) -> str:
    return f'{value:.{precision}f}'


def get_mb_str(byte_count: int) -> str:
    return format_float_str(byte_count / (1024 ** 2))
