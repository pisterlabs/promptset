import openai
import logging
import os
from .ai import OpenAI
from .error import FileSizeError, VAError

logger = logging.getLogger("chatty")

class OpenAIAudio(OpenAI):
    """
    Given byte limit for audio files for openai at the time of writing this code (25Mb)
    Supported audio file formats at the time of writing this code: ['m4a', 'mp3', 'webm', 'mp4', 'mpga', 'wav', 'mpeg']
    """
    BYTE_LIMIT:int = 26_214_400
    TEXT:str = "text"

    def __init__(self, model:str="whisper-1"):
        super().__init__(model)

    def transcribe(self, file:str) -> str:
        file = self.__open_file(file)
        response = self.__send_request(file, openai.Audio.transcribe)
        return response[self.TEXT]

    def translate(self, file:str) -> str:
        file = self.__open_file(file)
        response = self.__send_request(file, openai.Audio.translate)
        return response[self.TEXT]

    def __send_request(self, file, function):
        try:
            return function(self.model, file)
        except openai.OpenAIError as err:
            logger.error(err.json_body)
            raise VAError(err.json_body)

    def __open_file(self, file:str):
        try:
            self.__validate_size(file)
            return open(file, "rb")
        except (FileSizeError, FileNotFoundError) as err:
            logger.error(err.message)
            raise VAError(err.message)

    def __validate_size(self, file:str):
        try:
            size = os.path.getsize(file)
            if size >= self.BYTE_LIMIT:
                raise FileSizeError(f"Given file size {size} is larger than the limit {self.BYTE_LIMIT}")
        except OSError as err:
            logger.error(err)
            raise VAError(err)
