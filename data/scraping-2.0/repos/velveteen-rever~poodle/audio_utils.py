# Standard Library
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

# Third Party Libraries
import openai
import pyaudio
import simpleaudio
import torch.cuda
import vosk
import wave
from datasets import load_dataset
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline

# Local Modules
import config
import event_flags as ef
import whisper
from file_manager import FileManager
import soundfile


def playMp3Sound(file):
    sound = AudioSegment.from_mp3(file)
    sound_thread = threading.Thread(target=play, args=(sound,))
    sound_thread.start()


class KeywordDetector(threading.Thread):
    def __init__(self, keyword, audio_params=None, max_listener_threads=10):
        threading.Thread.__init__(self)
        self.stream_write_time = None
        self.keyword = keyword
        self.vosk = vosk
        self.vosk.SetLogLevel(-1)
        self.model = vosk.Model(lang="en-us")
        self.audio_queue = queue.Queue()
        self.keyword_listeners = []
        self.partial_listeners = []
        self.running = threading.Event()
        self.running.set()
        self.stream_write_time = None
        self.audio_params = audio_params or {}
        self.fetcher = AudioQueueFetcher(
            self.audio_queue, self.running, **self.audio_params
        )
        self.sample_rate = self.fetcher.sample_rate
        self.executor = ThreadPoolExecutor(max_listener_threads)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)

    def run(self):
        self.fetcher.start()
        try:
            while self.running.is_set() or not self.audio_queue.empty():
                self.stream_write_time, data = self.audio_queue.get()
                if self.recognizer.AcceptWaveform(data):
                    final_result = self.recognizer.FinalResult()
                    if self.keyword in final_result:
                        self.notify_keyword_listeners(data, self.stream_write_time)
                partial_result = self.recognizer.PartialResult()
                if partial_result:
                    self.notify_partial_listeners(partial_result)
                time.sleep(0.1)
            self.executor.shutdown()
        except queue.Empty:
            logging.error("Queue is empty.")
        except Exception as e:
            logging.critical(f"An error occurred: {e}")

    def add_keyword_listener(self, listener_func):
        self.keyword_listeners.append(listener_func)

    def notify_keyword_listeners(self, data, stream_write_time):
        for listener in self.keyword_listeners:
            self.executor.submit(listener, self.keyword, data, stream_write_time)

    def add_partial_listener(self, listener_func):
        self.partial_listeners.append(listener_func)

    def notify_partial_listeners(self, data):
        for listener in self.partial_listeners:
            self.executor.submit(listener, data)

    def close(self):
        self.running.clear()
        self.fetcher.stop()  # use the stop method of AudioFetcher


class AudioQueueFetcher(threading.Thread):
    def __init__(
        self,
        audio_queue,
        running,
        channels=config.PYAUDIO_CHANNELS,
        frames_per_buffer=8000,
    ):
        threading.Thread.__init__(self)
        self.audio_queue = audio_queue
        self.running = running
        self.pa = pyaudio.PyAudio()
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.default_device_info = self.pa.get_default_input_device_info()
        self.sample_rate = int(self.default_device_info["defaultSampleRate"])

    def run(self):
        stream = self.pa.open(
            format=config.PYAUDIO_FORMAT,
            channels=config.PYAUDIO_CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=config.PYAUDIO_FRAMES_PER_BUFFER,
        )
        while self.running.is_set():
            data = stream.read(self.frames_per_buffer)
            self.audio_queue.put((time.time(), data))
        stream.stop_stream()
        stream.close()
        self.pa.terminate()

    def stop(self):
        self.running.clear()


class Transcriber:
    def __init__(self, audio_directory, transcription_directory):
        self.audio_directory = audio_directory
        self.transcription_directory = transcription_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = "base"
        self.mod = whisper.load_model(self.model, self.device)

    def transcribe_bodies(self):
        while len(os.listdir(self.audio_directory)) != 0:
            for file in os.listdir(self.audio_directory):
                t = time.time()
                result = self.mod.transcribe(audio=f"{self.audio_directory}{file}")
                os.remove(f"{self.audio_directory}{file}")
                file = file.rstrip(".wav")
                FileManager.save_json(
                    f"{self.transcription_directory}transcription_{file}.json", result
                )
                logging.info(
                    f"transcription completed in: "
                    f"{time.time() - t} seconds using device: {self.device}, model: {self.model}\n"
                )


class OnlineTranscriber:
    def __init__(self, audio_directory, transcription_directory):
        self.audio_directory = audio_directory
        self.transcription_directory = transcription_directory
        self.ai = openai
        self.api_key = FileManager.read_file("api_keys/keys")

    def online_transcribe_bodies(self):
        if len(os.listdir(self.audio_directory)) != 0:
            for file in os.listdir(self.audio_directory):
                t = time.time()
                f = open(f"{self.audio_directory}{file}", "rb")
                result = self.ai.audio.transcriptions.create(
                    model="whisper-1", file=f, response_format="verbose_json"
                )
                os.remove(f"{self.audio_directory}{file}")
                file = file.rstrip(".wav")
                FileManager.save_json(
                    f"{self.transcription_directory}transcription_{file}.json", result
                )
                print(
                    f"transcription completed in: "
                    f"{time.time() - t} seconds using api call\n"
                )


class AudioRecorder:
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.default_device_info = self.pa.get_default_input_device_info()
        self.sample_rate = int(self.default_device_info["defaultSampleRate"])
        self.frames_per_buffer = 2048
        self.frames = []
        self.stream = None

    def start_recording(self):
        stream = self.pa.open(
            format=config.PYAUDIO_FORMAT,
            channels=config.PYAUDIO_CHANNELS,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
        )
        ef.recording.set()
        logging.info("recording started")
        self.frames.clear()
        while ef.recording.is_set():
            data = stream.read(self.frames_per_buffer)
            self.frames.append(data)
        self.stream = stream

    def stop_recording(self, filepath):
        ef.recording.clear()
        self.stream.close()
        self.stream.stop_stream()
        sound_file = wave.open(filepath, "wb")
        sound_file.setnchannels(config.PYAUDIO_CHANNELS)
        sound_file.setsampwidth(self.pa.get_sample_size(pyaudio.paInt16))
        sound_file.setframerate(self.sample_rate)
        sound_file.writeframes(b"".join(self.frames))
        logging.info("recording saved")


class SilenceWatcher:
    def __init__(self, silence_threshold=12, silence_duration=1.7):
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.silence_counter = 0
        self.silence_start_time = None

    def check_silence(self, pr) -> bool:
        no_speech = all(pr[key] == "" for key in pr)
        if no_speech:
            self.silence_counter += 1
            if self.silence_counter >= self.silence_threshold:
                if not self.silence_start_time:
                    self.silence_start_time = time.time()
                elif time.time() - self.silence_start_time >= self.silence_duration:
                    logging.info("silence detected")
                    return True
        else:
            self.reset()

    def reset(self):
        self.silence_counter = 0
        self.silence_start_time = None


class TextToSpeech:
    def __init__(self):
        self.model = "tts-1"
        self.tts = openai.audio
        self.playback_thread = None
        self.stop_playback = threading.Event()

    def stream_voice(self, text, voice):
        response = self.tts.speech.create(
            model=self.model, voice=voice, input=text, response_format="mp3"
        )
        audio_data = BytesIO(response.content)
        audio_segment = AudioSegment.from_file(audio_data, format="mp3")
        pcm_data = audio_segment.raw_data

        self.playback_thread = threading.Thread(
            target=self._play_audio,
            args=(
                pcm_data,
                audio_segment,
            ),
        )
        self.playback_thread.start()

    def _play_audio(self, pcm_data, audio_segment):
        play_obj = simpleaudio.play_buffer(
            pcm_data,
            num_channels=audio_segment.channels,
            bytes_per_sample=audio_segment.sample_width,
            sample_rate=audio_segment.frame_rate,
        )
        while play_obj.is_playing():
            if self.stop_playback.is_set():
                play_obj.stop()
                break

    def stop_audio(self):
        self.stop_playback.set()
        if self.playback_thread:
            self.playback_thread.join()

    def is_audio_playing(self):
        return self.playback_thread is not None and self.playback_thread.is_alive()


class TextToSpeechLocal:
    def __init__(self):
        self.synthesiser = pipeline("text-to-speech", model="microsoft/speecht5_tts")
        embeddings_dataset = load_dataset(
            "Matthijs/cmu-arctic-xvectors", split="validation"
        )
        self.speaker_embedding = torch.tensor(
            embeddings_dataset[7306]["xvector"]
        ).unsqueeze(0)
        self.play_obj = None
        self.playback_thread = None

    def generate_speech(self, text, file_name="speech.wav"):
        speech = self.synthesiser(
            text, forward_params={"speaker_embeddings": self.speaker_embedding}
        )
        soundfile.write(file_name, speech["audio"], samplerate=speech["sampling_rate"])
        return file_name

    def play_audio(self, file_name):
        if self.playback_thread and self.playback_thread.is_alive():
            self.stop_audio()  # Stop any currently playing audio
        self.playback_thread = threading.Thread(
            target=self._play_audio_thread, args=(file_name,)
        )
        self.playback_thread.start()

    def _play_audio_thread(self, file_name):
        wave_obj = simpleaudio.WaveObject.from_wave_file(file_name)
        self.play_obj = wave_obj.play()
        self.play_obj.wait_done()

    def stop_audio(self):
        if self.play_obj:
            self.play_obj.stop()
        if self.playback_thread:
            self.playback_thread.join()
