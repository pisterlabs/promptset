
import os
from google.cloud import speech
import logging
from dotenv import load_dotenv
import openai
from time import perf_counter
from whispercpp import Whisper
import whispercpp
import sys
from contextlib import contextmanager
import subprocess
import typing
import yaml

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Speech2Transcription:
    def __init__(
            self,
    ):
        raise RuntimeError(
            f'{self.__class__.__name__} class cannot be instantiated. Use the static methods instead.'
        )

    @staticmethod
    def _whisper_local_audio_preprocess(input_file: os.PathLike) -> bool:
        """
        Preprocess the audio file to be in the LINEAR16 format.
        :param input_file: The path to the audio file.
        :type input_file: os.PathLike
        :return: True if the preprocessing was successful, False otherwise.
        """
        try:
            subprocess.check_call(['./whisper_convert.sh', input_file])
            return True
        except subprocess.CalledProcessError:
            logging.error("Error during audio preprocessing.")
            exit(1)

    @staticmethod
    @contextmanager
    def _supress_prints():
        original = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        yield
        sys.stdout = original

    @classmethod
    def google_file_transcription(cls) -> speech.RecognizeResponse:
        """
        Transcription of audio file in LINEAR16 format using Google Cloud Speech-to-Text API.
        Long format files are stored in the Google Cloud Storage bucket.

        :param gc_uri: The path to the audio file in the Google Cloud Storage bucket.
        :type gc_uri: str
        :return: The transcript of the audio file.
        """
        google_client = speech.SpeechClient.from_service_account_file('keys.json')
        google_gc_uri = os.getenv("GC_URI")
        audio = speech.RecognitionAudio(uri=google_gc_uri)
        google_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="es-CO",
            enable_automatic_punctuation=True,
        )
        start_time = perf_counter()
        operation = google_client.long_running_recognize(config=google_config, audio=audio)
        logging.info("Waiting for the Cloud STT api to retrieve the transcript...")
        response = operation.result(timeout=90)
        logging.info(f"Transcript retrieved in {perf_counter() - start_time} seconds\n\n")

        transcript_builder = []
        for result in response.results:
            transcript_builder += [f"{result.alternatives[0].transcript}"]

        return "".join(transcript_builder)

    @classmethod
    def openai_file_transcription(
            cls,
            audio_file: os.PathLike,
            language: str = 'en',
            prompt_guidance: str = None,
        ) -> str:
        """
        Transcription of audio file using the OPEN AI API.
        The model used here is whisper-1.

        :param model: Whisper-1 by default no oder model is available
        :type model: str
        :param audio_file: the audio file to be transcribed
        :type audio_file: os.PathLike
        :param prompt_guidance: A textual prompt to help the model understand the context of the audio file
        :type prompt_guidance: str
        :return: Transcript of the audio file
        """
        logging.info(f'OPENAI API is running for transcription...\n\n')
        openai.api_key = os.getenv("OPENAI_API_KEY")
        audio = open(audio_file, 'rb')
        start_time = perf_counter()
        transcript = openai.Audio.transcribe(
            model='whisper-1',
            file=audio,
            language=language,
            temperature=0.3,
            prompt=prompt_guidance,
        )
        logging.info(f'Time taken by OPENAI Whisper-1: {perf_counter() - start_time} seconds.')
        return '\n'.join(transcript['text'].split('.'))

    @classmethod
    def local_whisper_transcribe_file(
            cls,
            model_path: os.PathLike,
            audio_file: os.PathLike,
            language: str = 'en',
    ) -> str:
        """
        Local inference of whisper model.
        :param model_path: The path where the model is stored
        :param audio_file: the audio file to be transcribed
        :param language: The language used in the audio file
        :return: A transcript.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'{model_path} does not exist.')

        if not cls._whisper_local_audio_preprocess(audio_file):
            raise RuntimeError('Error during audio preprocessing.')


        whisper = Whisper.from_pretrained(
            model_name=model_path
        )

        whisper.params.with_language(language)
        start_time = perf_counter()
        output = whisper.transcribe_from_file(os.path.join('audio_data', 'output_16.wav'))

        logging.info(f'Time taken by Local Whisper: {perf_counter() - start_time} seconds.')
        return '\n'.join(output.split('.'))

    @classmethod
    def stream_transcription_local_whisper(
            cls,
            model_name: os.PathLike,
            sample_rate: int = whispercpp.api.SAMPLE_RATE,
            language: str = 'es',

    ):
        iterator: typing.Iterator[str] | None = None
        with open(os.getenv('STREAM_CONFIG'), 'r') as stream: params = yaml.safe_load(stream)
        params['sample_rate'] = sample_rate
        params['language'] = language
        print(params)
        try:
            iterator = whispercpp.Whisper.from_pretrained(
                model_name=model_name
            ).stream_transcribe(**params)
        finally:
            assert iterator is not None, "Something went wrong!"
            sys.stderr.writelines(
                ["\nTranscription (line by line):\n"] + [f"{it}\n" for it in iterator]
            )
            sys.stderr.flush()




if __name__ == '__main__':
    # print(STT.local_whisper_transcribe_file(
    #     model_path='models/ggml-small.bin',
    #     audio_file='audio_data/audio_sample_2.wav',
    #     language='es'
    # ))
    STT.stream_transcription_local_whisper(
        model_name='models/ggml-small.bin',
    )

    # print(STT.openai_file_transcription(
    #     audio_file='audio_data/audio_sample_2.wav',
    #     language='es',
    #     prompt_guidance='Tienes que detectar quien habla en el audio.'
    # ))