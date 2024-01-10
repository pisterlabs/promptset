import http
import mimetypes
import tempfile

import fastapi
import magic
import openai
import pydantic

import lib.app.settings as app_settings
import lib.stt as stt


class OpenaiSpeechRepository:
    def __init__(self, settings: app_settings.Settings):
        self.settings = settings
        openai.api_key = self.settings.openai.api_key.get_secret_value()

    @staticmethod
    def __get_file_extension_from_bytes(audio: bytes) -> str | None:
        mime: magic.Magic = magic.Magic(mime=True)
        mime_type: str = mime.from_buffer(audio)
        extension: str | None = mimetypes.guess_extension(mime_type)
        if extension:
            extension = extension.replace(".", "")
        return extension

    async def speech_to_text(self, audio: bytes) -> str:
        file_extension = self.__get_file_extension_from_bytes(audio)
        if not file_extension or file_extension not in self.settings.voice.available_formats:
            raise fastapi.HTTPException(
                status_code=http.HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                detail=f"File extension is not supported. "
                f"Available extensions: {self.settings.voice.available_formats}",
            )
        try:
            voice: stt.models.SttVoice = stt.models.SttVoice(
                audio_size=len(audio) // 1024,  # audio size in MB,
                audio_format=file_extension,
                audio_data=audio,
                voice_settings=self.settings.voice,
            )
        except (pydantic.ValidationError, ValueError) as e:
            raise fastapi.HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=f"Voice validation error: {e}",
            )

        try:
            with tempfile.NamedTemporaryFile(suffix=f".{file_extension}") as temp_file:
                temp_file.write(voice.audio_data)
                temp_file.seek(0)
                transcript = openai.Audio.transcribe(self.settings.openai.stt_model, temp_file)  # type: ignore
        except openai.error.InvalidRequestError as e:  # type: ignore[reportGeneralTypeIssues]
            raise fastapi.HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=f"OpenAI request error: {e}",
            )
        except openai.error.OpenAIError as e:  # type: ignore[reportGeneralTypeIssues]
            raise fastapi.HTTPException(
                status_code=http.HTTPStatus.BAD_REQUEST,
                detail=f"OpenAI API error: {e}",
            )

        return transcript.text  # type: ignore[reportUnknownVariableType]
