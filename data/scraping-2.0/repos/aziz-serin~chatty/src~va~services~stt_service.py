from werkzeug.datastructures import FileStorage
from flask import Response, json
from .service import Service
from src.va.openai_tools.ai_audio import OpenAIAudio
from src.va.openai_tools.error import VAError
import logging
import os

class SttService(Service):
    ALLOWED_EXTENSIONS:list = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}
    TRANSCRIBE:str = "transcribe"
    TRANSLATE:str = "translate"
    ALLOWED_STT_MODELS = "whisper-1"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("chatty")

    def stt(self, filename:str, file:FileStorage, form:dict, method:str):
        model = self.__handle_form__(form)
        if model is None:
            audio = OpenAIAudio()
        else:
            audio = OpenAIAudio(model=model)
        if method == self.TRANSCRIBE:
            return self.transcribe(filename, file, audio)
        elif method == self.TRANSLATE:
            return self.translate(filename, file, audio)
        else:
            self.logger.debug(f"Method '{method}' is unknown for stt")
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )
    def transcribe(self, filename:str, file:FileStorage, audio:OpenAIAudio) -> Response:
        return self.__speech_to_text__(filename, file, audio.transcribe)

    def translate(self, filename:str, file:FileStorage, audio:OpenAIAudio) -> Response:
        return self.__speech_to_text__(filename, file, audio.translate)

    def __speech_to_text__(self, filename:str, file:FileStorage, function) -> Response:
        path = self.__save_file__(filename, file)
        if path is None:
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )

        try:
            text = function(path)
            self.__handle_delete__(path)
            return Response(
                response=json.dumps({
                    "response":text
                }),
                status=200,
                mimetype='application/json'
            )
        except VAError as err:
            self.logger.debug(err)
            self.__handle_delete__(path)
            return Response(
                response=json.dumps({
                    "reason": "Internal Server Error"
                }),
                status=500,
                mimetype='application/json'
            )

    def allowed_file(self, filename:str) -> bool:
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in self.ALLOWED_EXTENSIONS

    def __save_file__(self, filename:str, file:FileStorage) -> str | None:
        try:
            path = os.path.join(self.filepath, filename)
            file.save(path)
            return path
        except IOError:
            self.logger.error(f"Could not save the file {filename} to specified path {self.filepath}")
            return None
        finally:
            file.close()

    def __handle_delete__(self, filepath:str):
        try:
            os.remove(filepath)
        except OSError as err:
            self.logger.error(err)
            raise err

    def __handle_form__(self, form:dict) -> str | None:
        if "stt_model" not in form:
            return None
        if form["stt_model"] not in self.ALLOWED_STT_MODELS:
            return None
        return form["stt_model"]
