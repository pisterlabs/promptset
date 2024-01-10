import mimetypes
import os
from pathlib import Path
from pprint import pprint
from typing import Optional, Literal
from urllib.parse import urlparse

import aiofiles
import aiohttp
from moviepy.editor import VideoFileClip
from openai import AsyncOpenAI
from pydub import AudioSegment

from jonbot.backend.data_layer.models.voice_to_text_request import VoiceToTextResponse
from jonbot.system.path_getters import get_temp_folder
from jonbot.system.setup_logging.get_logger import get_jonbot_logger

logger = get_jonbot_logger()


async def download_file(session, url, destination_path):
    async with session.get(url) as response:
        response.raise_for_status()  # Will raise an exception for non-200 responses
        async with aiofiles.open(destination_path, mode="wb") as file:
            await file.write(await response.read())


def convert_to_mp3(original_path, target_path):
    mimetype = mimetypes.guess_type(original_path)[0]
    if 'video' in mimetype:
        with VideoFileClip(str(original_path)) as video:
            video.audio.write_audiofile(str(target_path))
    elif 'audio' in mimetype:
        audio = AudioSegment.from_file(str(original_path), format=original_path.suffix.lstrip('.'))
        audio.export(target_path, format="mp3")
    else:
        raise ValueError(f"Unsupported file format: {mimetype}")


async def send_transcription_request(audio_path: str,
                                     model: str = "whisper-1",
                                     prompt: str = None,
                                     temperature: float = 0,
                                     response_format: Literal[
                                         "json", "text", "srt", "verbose_json", "vtt"] = "verbose_json",
                                     language: Optional[str] = None) -> dict:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(audio_path, "rb") as audio_file:
        response = await client.audio.transcriptions.create(file=audio_file,
                                                            model=model,
                                                            prompt=prompt,
                                                            response_format=response_format,
                                                            temperature=temperature,
                                                            language=language)
    if not response:
        raise Exception("Transcription request returned None.")
    return response


async def transcribe_audio(audio_source: str,
                           model: str = "whisper-1",
                           prompt: str = None,
                           temperature: float = 0,
                           response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "verbose_json",
                           language: Optional[str] = None,
                           **kwargs,
                           ) -> VoiceToTextResponse:
    parsed_source = urlparse(audio_source)
    is_url = parsed_source.scheme in ('http', 'https')
    temp_folder = get_temp_folder()

    audio_file_name =  f"audio-file{Path(audio_source).suffix}"
    audio_file_name = audio_file_name.split("?")[0]
    temp_original_path = Path(temp_folder) / audio_file_name
    temp_mp3_path = Path(temp_folder) / "audio-file.mp3"

    async with aiohttp.ClientSession() as session:
        if is_url:
            await download_file(session, audio_source, temp_original_path)
        elif not Path(audio_source).exists():
            raise FileNotFoundError(f"Audio file not found at {audio_source}")
        else:
            temp_original_path = Path(audio_source)

        convert_to_mp3(temp_original_path, temp_mp3_path)
        response = await send_transcription_request(audio_path=str(temp_mp3_path),
                                                    model=model,
                                                    prompt=prompt,
                                                    temperature=temperature,
                                                    response_format=response_format,
                                                    # language=language
                                                    )
    # delete temp files
    if is_url:
        # delete downloaded file if it was a url, otherwise it was a local file and we don't want to delete it
        os.remove(temp_original_path)
    os.remove(temp_mp3_path)
    return VoiceToTextResponse(text=response.text, success=True, mp3_file_path=str(temp_mp3_path),
                               metadata=response.dict())


if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    load_dotenv()

    response = asyncio.run(
        transcribe_audio(
            audio_source=r"C:\Users\jonma\Downloads\5294aef968868b1b82e4a3627712a259.mp4",
            prompt="This is a transcript from Bisam in Ghaza, Palestine",
            response_format="verbose_json",
            temperature=0.9,
            language="en-US",
        )
    )

    pprint(response, indent=4)
