from app.services.resource_processors.ai_resource_processor import AiResourceProcessor
from app.services.resource_processors.captions_resource_processor import CaptionsResourceProcessor
from app.services.resource_processors.hybrid_resource_processor import HybridResourceProcessor
from app.services.speech_recognizers.assembly_ai_speech_recognizer import AssemblyAiSpeechRecognizer
from app.services.speech_recognizers.gladia_speech_recognizer import GladiaSpeechRecognizer
from app.services.speech_recognizers.google_speech_recognizer import GoogleSpeechRecognizer
from app.services.speech_recognizers.microsoft_azure_speech_recognizer import MicrosoftAzureSpeechRecognizer
from app.services.speech_recognizers.open_ai_whisper_speech_recognizer import OpenAIWhisperSpeechRecognizer


class RecognizerData:
    RECOGNIZER_TYPE_TO_CLASS = {
        'assemblyai':  AssemblyAiSpeechRecognizer,
        'gladia': GladiaSpeechRecognizer,
        'google':  GoogleSpeechRecognizer,
        'microsoft':  MicrosoftAzureSpeechRecognizer,
        'openai':  OpenAIWhisperSpeechRecognizer
    }

    PROCESSOR_TYPE_TO_CLASS = {
        'captions':  CaptionsResourceProcessor,
        'hybrid': HybridResourceProcessor,
        'ai':  AiResourceProcessor,
    }

    def __init__(self, language_code, recognizer='google', captions=False):
        self.language_code = language_code
        self.recognizer_class = RecognizerData.RECOGNIZER_TYPE_TO_CLASS[recognizer or 'google']
        self.processor_class = RecognizerData.PROCESSOR_TYPE_TO_CLASS[RecognizerData._processor_requested(
            captions)]

    def __str__(self):
        attributes_str = ''
        for item in self.__dict__:
            item_str = '{} = {}'.format(item, self.__dict__[item])
            attributes_str += '\n' + item_str
        return str(self.__class__) + '\n' + attributes_str

    def __eq__(self, other):
        if isinstance(other, RecognizerData):
            return self.recognizer_class == other.recognizer_class \
                and self.language_code == other.language_code \
                and self.processor_class == other.processor_class
        return False

    @staticmethod
    def _processor_requested(captions):
        if captions == True:
            return 'captions'
        elif captions == 'try':
            return 'hybrid'
        else:
            return 'ai'
