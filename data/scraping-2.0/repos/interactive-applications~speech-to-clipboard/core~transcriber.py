import os
import openai
from openai.openai_object import OpenAIObject


class Transcriber:
    
    def __init__(
        self,
        openai_api_key: str = "",
        use_local: bool = False,
        use_api: bool = True,
        whisper_model: str = "base"
    ) -> None:
        """
        Initialize the Transcriber class.
        
        :param openai_api_key: OpenAI API key
        """
        openai.api_key = openai_api_key
        
        self.model = None
        self.use_local = use_local
        self.use_api = use_api
        
        if use_local:
            try:
                import whisper #pylint: disable=import-outside-toplevel
                self.model = whisper.load_model(whisper_model)
            except ImportError:
                pass
    
    def get_file_to_transcribe(
        self, wav_file_path: str = "./audio/output.wav"
    ) -> str:
        """
        Get the file to transcribe.
        Tries to find a file with the same name as the given file, but with a different extension.
        If no such file exists, the given file is returned.
        
        :wav_file_path: The path to the wav file to transcribe.
        :return: The path to the file to transcribe.
        """
        file_path = wav_file_path
        base_path = os.path.splitext(wav_file_path)[0]
        base_dir = os.path.dirname(base_path)
        basename = os.path.basename(base_path)
        
        for file in os.listdir(base_dir):
            if os.path.splitext(file)[0] == basename:
                file_path = f"{base_dir}/{file}"
                return file_path
        
        return wav_file_path
    
    def transcribe_locally(self, file_path: str = "./audio/output.wav") -> str:
        """
        Try to transcribe the given wav file locally.
        
        :param file_path: The path to the wav file to transcribe.
        :return: The transcription of the wav file.
        """
        
        if self.model is not None:
            result = self.model.transcribe(file_path)
            transcript = result['text']
            return transcript
        
        raise ModuleNotFoundError(
            "Unable to transcribe locally. Whisper not installed?"
        )
    
    def transcribe_via_api(self, file_path: str = "./audio/output.wav") -> str:
        """
        Transcribe the given wav file via the OpenAI API.
        
        :param file_path: The path to the wav file to transcribe.
        :return: The transcription of the wav file.
        """
        
        with open(file_path, "rb") as audio_file:
            transcript: OpenAIObject = openai.Audio.transcribe(
                "whisper-1", audio_file
            )
            transcript = transcript['text']
        return transcript
    
    def transcribe(
        self,
        wav_file_path: str = "./audio/output.wav",
        delete_file_on_success: bool = True
    ) -> str:
        """
        Trnascribe the given wav file.
        
        :param wav_file_path: The path to the wav file to transcribe.
        :return: The transcription of the wav file.
        """
        
        file_path = self.get_file_to_transcribe(wav_file_path)
        
        if self.use_local:
            try:
                transcript = self.transcribe_locally(file_path)
                
                if delete_file_on_success:
                    os.remove(file_path)
                
                return transcript
            except Exception as e: #pylint: disable=broad-except
                print(f"Local transcription failed: \n{e}")
                if self.use_api:
                    print("Falling back to API transcription...")
        
        if not self.use_local or self.use_api:
            try:
                transcript = self.transcribe_via_api(file_path)
                
                if delete_file_on_success:
                    os.remove(file_path)
                
                return transcript
            except Exception as e: #pylint: disable=broad-except
                return "Error: " + str(e)
