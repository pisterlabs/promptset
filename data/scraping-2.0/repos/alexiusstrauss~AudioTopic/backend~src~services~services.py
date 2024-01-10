from uuid import uuid4

import speech_recognition as sr
from fastapi import Request, UploadFile
from gtts import gTTS
from pydub import AudioSegment as Asegment

from src.services.exceptions import RecognizeException, SummarizeException
from src.summarization.engines import LangChain
from src.summarization.interfaces import Summarization


class DeepDive:
    """
    Service responsavel por upload, conversao e validação do audio
    """

    def __init__(self, llm_engine: Summarization):
        self.llm_engine = llm_engine

    def upload_audio(self, file: UploadFile):
        """
        Function upload audio mp3 or wav
        """
        file_extension = file.filename.split(".")[-1]
        if file_extension not in ["mp3", "wav"]:
            return False

        file_id = str(uuid4())
        file_location = f"audio_files/{file_id}.{file_extension}"

        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

        return {"file_id": file_id, "file_location": file_location}

    def speech_to_text(self, upload_audio: dict) -> dict:
        """
        Function convert audio to text
        """
        recognizer = sr.Recognizer()
        audio_file_path = upload_audio["file_location"]
        audio_file_path = self.__convert_mp3_to_wav(audio_file_path)

        with sr.AudioFile(audio_file_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data=audio_data, language="pt-BR")
                upload_audio["original_context"] = text
            except (sr.UnknownValueError, sr.RequestError) as exc:
                raise RecognizeException() from exc

        return upload_audio

    def __convert_mp3_to_wav(self, mp3_file_path):
        wav_file_path = mp3_file_path.replace(".mp3", ".wav")
        try:
            Asegment.from_mp3(mp3_file_path).export(wav_file_path, format="wav")
        except Exception as exc:
            raise RecognizeException() from exc
        return wav_file_path

    def summarize_text(self, response: dict) -> str:
        try:
            text_to_sumarize = response.get("original_context")
            result = self.llm_engine.summarize(text=text_to_sumarize)
            response["summary_context"] = result
            return response
        except Exception as exc:
            print(f"Erro na sumazicação do texto: {exc}")
            raise SummarizeException() from exc

    def validate_api_token(self):
        if isinstance(self.llm_engine, LangChain):
            self.llm_engine.token_is_valid()

    def create_audio_from_summary(self, response: dict):
        """
        Function to create an audio file from the summarized text
        """
        summarized_text = response.get("summary_context")
        if not summarized_text:
            raise ValueError("No summarized text found in the response")

        tts = gTTS(text=summarized_text, lang="pt-br", slow=False)
        audio_file_path = f"audio_files/{response['file_id']}_audio_summarize.mp3"

        try:
            tts.save(audio_file_path)
            response["mp3_summary_url"] = audio_file_path
            return response
        except Exception as exc:
            print(f"Erro ao criar arquivo de áudio: {exc}")
            raise

    def create_link_to_summary(self, request: Request, response: dict):
        base_url = str(request.base_url)
        response["summary_url"] = f"{base_url}download/{response.get('file_id')}"
        return response
