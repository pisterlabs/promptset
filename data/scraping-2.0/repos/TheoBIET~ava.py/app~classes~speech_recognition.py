"""Class used to transcribe the user voice into text."""
import whisper

from app.classes.openai_requester import OpenAI
from app.config.speech_recognition import SPEECH_RECOGNITION

class SpeechRecognition:            # pylint: disable=too-few-public-methods
    """Class used to transcribe the user voice into text."""
    def __init__(self):
        self._use_local = SPEECH_RECOGNITION['USE_LOCAL']
        self._lang = SPEECH_RECOGNITION['LANGUAGE']

        if self._use_local:
            print('🤫 Loading Whisper Model')
            self._options = whisper.DecodingOptions(fp16=SPEECH_RECOGNITION['LOCAL']['FP16'],
                                                    language=self._lang)
            self._model = whisper.load_model(SPEECH_RECOGNITION['LOCAL']['MODEL'],
                                             device=SPEECH_RECOGNITION['LOCAL']['DEVICE'])
        else:
            print('🤫 Loading Whisper API')
            self._openai = OpenAI()

    def transcribe(self, file):
        """Transcribe an audio file to text using whisper."""
        if self._use_local:
            return self._model.transcribe(file, language=self._lang)['text']

        return self._openai.transcribe(file)
