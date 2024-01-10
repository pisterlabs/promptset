import io
import queue
import wave

import numpy as np
import whisper

from sic_framework import SICComponentManager, SICConfMessage
from sic_framework.core.component_python2 import SICComponent
from sic_framework.core.connector import SICConnector
from sic_framework.core.message_python2 import SICMessage, SICRequest, AudioMessage

import openai
import speech_recognition as sr


class WhisperConf(SICConfMessage):
    """
    Provide either an openai key or a model name to run locally
    :param openai_key: your secret OpenAI key, see https://platform.openai.com/docs/quickstart
    :param local_model: Local OpenAI model to use, see https://github.com/openai/whisper#available-models-and-languages
    """

    def __init__(self, openai_key=None, local_model="base.en"):
        super(SICConfMessage, self).__init__()
        self.openai_key = openai_key
        self.model = local_model

class GetTranscript(SICRequest):
    def __init__(self, timeout=None, phrase_time_limit=None):
        """
        The ``timeout`` parameter is the maximum number of seconds that this will wait for a phrase to start before giving up and throwing an ``speech_recognition.WaitTimeoutError`` exception. If ``timeout`` is ``None``, there will be no wait timeout.
        The ``phrase_time_limit`` parameter is the maximum number of seconds that this will allow a phrase to continue before stopping and returning the part of the phrase processed before the time limit was reached. The resulting audio will be the phrase cut off at the time limit. If ``phrase_timeout`` is ``None``, there will be no phrase time limit.
        """
        super().__init__()

        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit


class Transcript(SICMessage):
    """
    """
    def __init__(self, transcript):
        super().__init__()
        self.transcript = transcript

class RemoteAudioDevice(sr.AudioSource):
    class Stream:
        def __init__(self):
            self.queue = queue.Queue()

        def clear(self):
            with self.queue.mutex:
                self.queue.queue.clear()

        def write(self, bytes):
            self.queue.put(bytes)
        def read(self, n_bytes):
            # todo check n_bytes equeals chunk_size
            return self.queue.get()

    def __init__(self, sample_rate=16000, sample_width=2, chunk_size = 2730):
        """
        This class imitates a pyaudio device to use the speech recoginizer API
        Default parameters are for NAO and Pepper
        """
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = sample_width

        self.CHUNK = chunk_size
        self.stream = self.Stream()



class WhisperComponent(SICComponent):
    """
    Dummy SICAction
    """
    COMPONENT_STARTUP_TIMEOUT = 5

    def __init__(self, *args, **kwargs):
        super(WhisperComponent, self).__init__(*args, **kwargs)

        # self.model = whisper.load_model("base.en")

        self.recognizer = sr.Recognizer()

        self.source = RemoteAudioDevice()

        self.parameters_are_inferred = False

        self.i = 0




    @staticmethod
    def get_inputs():
        return [AudioMessage, GetTranscript]

    @staticmethod
    def get_output():
        return Transcript


    @staticmethod
    def get_conf():
        return WhisperConf()


    def on_message(self, message):

        if not self.parameters_are_inferred:
            self.source.SAMPLE_RATE = message.sample_rate
            self.source.CHUNK = len(message.waveform)
            self.parameters_are_inferred = True
            self.logger.info("Inferred sample rate: {} and chunk size: {}".format(self.source.SAMPLE_RATE, self.source.CHUNK))

        self.source.stream.write(message.waveform)

    def on_request(self, request):
        self.source.stream.clear()
        print("Listening")
        audio = self.recognizer.listen(self.source, timeout=request.timeout, phrase_time_limit=request.phrase_time_limit)
        print("Transcribing")
        if self.params.openai_key:
            wav_data = io.BytesIO(audio.get_wav_data())
            wav_data.name = "SpeechRecognition_audio.wav"
            response = openai.Audio.transcribe("whisper-1", wav_data, api_key=self.params.openai_key, language="en", response_format="verbose_json")

        else:
            response = self.recognizer.recognize_whisper(audio, language="english", model=self.params.model, show_dict=True)

        print("FULL RESPONSE", response)
        transcript = response['text']

        no_speech_prob = np.mean([segment["no_speech_prob"] for segment in response['segments']])

        if no_speech_prob > .5:
            print("Whisper heard silence")
            return Transcript("")
        print("Whisper thinks you said: " + transcript)

        with wave.open(f"audio{self.i}.wav", 'wb') as f:
            f.setnchannels(1)
            f.setsampwidth(self.source.SAMPLE_WIDTH)  # number of bytes
            f.setframerate(self.source.SAMPLE_RATE)
            f.writeframesraw(audio.frame_data)
        self.i+=1

        return Transcript(transcript)






class SICWhisper(SICConnector):

    component_class = WhisperComponent


if __name__ == '__main__':
    # Request the service to start using the SICServiceManager on this device
    SICComponentManager([WhisperComponent])

