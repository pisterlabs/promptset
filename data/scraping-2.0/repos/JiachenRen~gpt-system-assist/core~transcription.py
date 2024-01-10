import io
import speech_recognition as sr

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import openai


class RealTimeTranscription:

    def __init__(self, record_timeout=2, phrase_timeout=3, transcription_timeout=2, prompt: str = 'listening...'):
        self.prompt = prompt
        self.data_queue = Queue()
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        self.source = None
        # How real time the recording is in seconds
        self.record_timeout = record_timeout
        # How much empty space between recordings before we consider it a new line in the transcription.
        self.phrase_timeout = phrase_timeout
        self.transcription_timeout = transcription_timeout
        self.temp_file = NamedTemporaryFile(suffix='.wav').name

    def record_callback(self, _, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        self.data_queue.put(data)

    def _init_audio_source(self):
        if 'linux' in platform:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                self.source = sr.Microphone(sample_rate=16000, device_index=index)
                break
        else:
            self.source = sr.Microphone(sample_rate=16000)

    def get_transcription(self):
        print(self.prompt, end='\r', flush=True)
        transcription = ['']
        last_sample = bytes()
        phrase_time = None
        self._init_audio_source()
        self.data_queue = Queue()

        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)

        # Create a background thread that will pass us raw audio bytes.
        # We could do this manually but SpeechRecognizer provides a nice helper.
        stop_listening = self.recorder.listen_in_background(
            self.source, self.record_callback, phrase_time_limit=self.record_timeout)

        while True:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if self.data_queue.empty():
                if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                    stop_listening(wait_for_stop=True)
                    return "\n".join(transcription)
            else:
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=self.phrase_timeout):
                    last_sample = bytes()
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now

                # Concatenate our current audio data with the latest audio data.
                while not self.data_queue.empty():
                    data = self.data_queue.get()
                    last_sample += data

                # Use AudioData to convert the raw data to wav data.
                audio_data = sr.AudioData(last_sample, self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH)
                wav_data = io.BytesIO(audio_data.get_wav_data())

                # Write wav data to the temporary file as bytes.
                with open(self.temp_file, 'w+b') as f:
                    f.write(wav_data.read())

                # Read the transcription.
                audio_file = open(self.temp_file, 'rb')
                result = openai.Audio.transcribe(
                    "whisper-1", audio_file, prompt="Do not hallucinate, only transcribe what you hear for certain.")
                text = result['text'].strip()

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                else:
                    transcription[-1] = text
                    print(text + " " * len(self.prompt), end='\r', flush=True)

                # Infinite loops are bad for processors, must sleep.
                sleep(0.25)


if __name__ == "__main__":
    t = RealTimeTranscription()
    while True:
        print(t.get_transcription())
