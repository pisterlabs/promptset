import os
import openai


class WhisperService:
    def __init__(self):
        self.__OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        self.__OPENAI_ORGANIZATION = os.environ.get("OPEN_AI_ORGANIZATION")
        openai.api_key = self.__OPENAI_KEY
        openai.organization = self.__OPENAI_ORGANIZATION

    def is_valid_tokens(self) -> bool:
        if self.__OPENAI_KEY and self.__OPENAI_ORGANIZATION:
            print("API key found in environment variable.")
            return True
        else:
            print("API key not found in environment variable.")
            return False

    @staticmethod
    def isValidAudioFile(filepath: str) -> bool:
        if not isinstance(filepath, str):
            return False
        fileSizeMB = os.path.getsize(filepath) / 1e6
        print(f"File received with {fileSizeMB} MB")
        if fileSizeMB > 10:
            print("File Size is larger than 10MB so process cancelled.")
            return False
        return True

    def transcriptAudioFile(self, filepath: str):
        if not WhisperService.isValidAudioFile(filepath):
            return False
        if not self.is_valid_tokens():
            return False

        audio_file = open(filepath, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format='text')
        return transcript['text']
