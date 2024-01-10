import pytest
from app.services.resource_processors.ai_resource_processor import AiResourceProcessor
from app.services.resource_processors.captions_resource_processor import CaptionsResourceProcessor
from app.services.resource_processors.hybrid_resource_processor import HybridResourceProcessor
from app.input_items.recognizer_data import RecognizerData
from app.services.speech_recognizers.assembly_ai_speech_recognizer import AssemblyAiSpeechRecognizer
from app.services.speech_recognizers.gladia_speech_recognizer import GladiaSpeechRecognizer
from app.services.speech_recognizers.google_speech_recognizer import GoogleSpeechRecognizer
from app.services.speech_recognizers.microsoft_azure_speech_recognizer import MicrosoftAzureSpeechRecognizer
from app.services.speech_recognizers.open_ai_whisper_speech_recognizer import OpenAIWhisperSpeechRecognizer


class TestRecognizerData:
    @pytest.mark.parametrize("recognizer, expected_class", [
        ('assemblyai', AssemblyAiSpeechRecognizer),
        ('gladia', GladiaSpeechRecognizer),
        ('google', GoogleSpeechRecognizer),
        ('microsoft', MicrosoftAzureSpeechRecognizer),
        ('openai', OpenAIWhisperSpeechRecognizer),
    ])
    def test_recognizer_class(self, recognizer, expected_class):
        recognizer_data = RecognizerData('en-US', recognizer)
        assert recognizer_data.recognizer_class == expected_class

    def test_default_recognizer_class(self):
        recognizer_data = RecognizerData('en-US')
        assert recognizer_data.recognizer_class == GoogleSpeechRecognizer

    def test_init_language_code(self):
        recognizer_data = RecognizerData('en-US')
        assert recognizer_data.language_code == 'en-US'

    def test_str_representation(self):
        recognizer_data = RecognizerData('en-US', 'google')
        expected_str = (
            f"<class '{RecognizerData.__module__}.{RecognizerData.__name__}'>\n\n"
            'language_code = en-US\n'
            "recognizer_class = <class 'app.services.speech_recognizers.google_speech_recognizer.GoogleSpeechRecognizer'>\n"
            "processor_class = <class 'app.services.resource_processors.ai_resource_processor.AiResourceProcessor'>"
        )
        assert str(recognizer_data) == expected_str

    def test_invalid_recognizer_type(self):
        with pytest.raises(KeyError):
            RecognizerData('en-US', 'invalid_recognizer')

    def test_eq_same_instance(self):
        recognizer_data1 = RecognizerData('en-US', 'google')
        assert recognizer_data1 == recognizer_data1

    def test_eq_different_instances_same_values(self):
        recognizer_data1 = RecognizerData('en-US', 'google')
        recognizer_data2 = RecognizerData('en-US', 'google')
        assert recognizer_data1 == recognizer_data2

    def test_eq_different_language_code(self):
        recognizer_data1 = RecognizerData('en-US', 'google')
        recognizer_data2 = RecognizerData('es-ES', 'google')
        assert recognizer_data1 != recognizer_data2

    def test_eq_different_recognizer_class(self):
        recognizer_data1 = RecognizerData('en-US', 'google')
        recognizer_data2 = RecognizerData('en-US', 'openai')
        assert recognizer_data1 != recognizer_data2

    def test_eq_different_class(self):
        recognizer_data = RecognizerData('en-US', 'google')
        assert recognizer_data != 'some_string'

    def test_eq_different_processor_class(self):
        recognizer_data1 = RecognizerData('en-US', 'google', captions=True)
        recognizer_data2 = RecognizerData('en-US', 'google', captions=False)
        assert recognizer_data1 != recognizer_data2

    @pytest.mark.parametrize("captions, expected_processor_class", [
        (True, CaptionsResourceProcessor),
        ('try', HybridResourceProcessor),
        (False, AiResourceProcessor),
    ])
    def test_processor_class(self, captions, expected_processor_class):
        recognizer_data = RecognizerData('en-US', captions=captions)
        assert recognizer_data.processor_class == expected_processor_class

    def test_default_processor_class(self):
        recognizer_data = RecognizerData('en-US')
        assert recognizer_data.processor_class == AiResourceProcessor

    def test_str_representation_with_captions_processor(self):
        recognizer_data = RecognizerData('en-US', 'google', captions=True)
        expected_str = (
            f"<class '{RecognizerData.__module__}.{RecognizerData.__name__}'>\n\n"
            'language_code = en-US\n'
            "recognizer_class = <class 'app.services.speech_recognizers.google_speech_recognizer.GoogleSpeechRecognizer'>\n"
            "processor_class = <class 'app.services.resource_processors.captions_resource_processor.CaptionsResourceProcessor'>"
        )
        assert str(recognizer_data) == expected_str
