from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from pynput import keyboard
import speech_recognition as sr
import pyaudio
import time
import os
import openai
import tempfile
import threading

class VoiceInput():
    def __init__(self, audio_player, voice_trigger=False):
        self.voice_trigger = voice_trigger
        self.recognizer = sr.Recognizer()
        self._keyboard_listening = False
        self.audio_player = audio_player
        self.listening = True
        self.low_volume = 0.1
        self.original_volume = 1
        self.devices = AudioUtilities.GetSpeakers()
        self.interface = self.devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None
        )
        self.volume_controller = cast(self.interface, POINTER(IAudioEndpointVolume))
        self.saving_volume = True
        self.whisper_thread_open = False
        self.transcribed_text = ""
        self.audio_data_queue = []
        self.recording_audio = False
        self.audio_chunks = []
        self.audio_chunk = None
        self.transcribing = False

        self.mic_thread_running = False
        self.detected_silence = False
        self.detected_index = []

        with sr.Microphone() as mic:
            self.SAMPLE_RATE = mic.SAMPLE_RATE
            self.SAMPLE_WIDTH = mic.SAMPLE_WIDTH
        

    def init(self, fn):
        if not self.voice_trigger:
            self._init_keyboard_listeners()

        self.listening = True

        while self.listening:
            while self._keyboard_listening:
                print('Listening...')
                text = self._start_record()

                if not text:
                    self._init_keyboard_listeners()
                    break

                print(f"{text} - it's a good read, sending to prompt")
                fn(text)

                while self.audio_player.is_playing or len(self.audio_player.queue) > 0:
                    time.sleep(0.3)
                    print('still waiting')

                self._init_keyboard_listeners()

            time.sleep(0.05)

    def _init_keyboard_listeners(self):
        def onpress(key):
            try:
                if key == keyboard.Key.f20:
                    self._keyboard_listening = True
                    if self.saving_volume:
                        self.original_volume = self.volume_controller.GetMasterVolumeLevelScalar()
                        self.volume_controller.SetMasterVolumeLevelScalar(self.low_volume, None)
                        self.saving_volume = False
            except:
                pass

        def onrelease(key):
            try:
                if key == keyboard.Key.f20:
                    self._keyboard_listening = False
                    self.saving_volume = True
                    return False
            except:
                pass

        listener = keyboard.Listener(on_press=onpress, on_release=onrelease)
        listener.start()

    def _start_record(self, buffer_size=1024):
        """
        This function records audio, transcribes voice input into text form, and returns the text.

        Parameters:
        buffer_size (int): The size of the audio buffer.

        Returns:
        str: The transcribed text from the voice input.
        """

        self.transcribing = True
        self.open_mic()

        while self._keyboard_listening:
            time.sleep(0.05)

        self.audio_chunks.clear()

        self.volume_controller.SetMasterVolumeLevelScalar(self.original_volume, None)

        if self.transcribing:
            time.sleep(0.05)
            pass

        temp_text = self.transcribed_text
        self.transcribed_text = ""

        return temp_text

    def whisper_transcribe(self):
        audio_data = self.audio_data_queue.pop(0)
        
        # Create an AudioData object from the raw data
        audio_obj = sr.AudioData(audio_data, self.SAMPLE_RATE, self.SAMPLE_WIDTH)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Write WAV data to the temp file
            temp_file.write(audio_obj.get_wav_data())
            temp_path = temp_file.name

        with open(temp_path, "rb") as audio_file:
            # Ensure that the file is not empty
            if os.path.getsize(temp_path) > 0:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            else:
                print("The audio file is empty.")

        os.remove(temp_path)
        self.transcribed_text += f" {transcript['text']}"
        print(transcript["text"])

        if len(self.audio_data_queue) > 0:
            self.whisper_transcribe()
        else:
            self.transcribing = False


    def data_to_queue(self, audio_chunks):
        audio_data = b''.join(audio_chunks)
        self.audio_data_queue.append(audio_data)
        self.whisper_transcribe()


    def mic_stream(self, stream, p):
        silence_start_time = None
        silent_called = False
        detected = False
        detected_index = [0]
        sound_detected = False
        audio_chunks = []

        while self._keyboard_listening:
            chunk = stream.read(1024)
            audio_chunks.append(chunk)

            values = [int.from_bytes(chunk[i:i+2], 'little', signed=True) for i in range(0, len(chunk), 2)]

            if max(values) > 700:
                sound_detected = True

            if max(values) < 300: 
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > 1.5 and not detected and sound_detected: 
                    detected = True
                    silent_called = True
                    index = len(audio_chunks)
                    detected_index.append(index)
                    t = threading.Thread(target=self.cut_audio_chunks, args = (detected_index, audio_chunks, ))
                    t.start()
                    
            else:
                detected = False
                silence_start_time = None
        try:
            t.join()
        except:
            pass

        print(sound_detected)
        stream.stop_stream()
        stream.close()
        p.terminate()

        if not silent_called and sound_detected:
            if self.has_sound(audio_chunks):
                self.transcribing = True
                audio_chunks = self.trim_data(audio_chunks)
                self.data_to_queue(audio_chunks)

        elif len(detected_index) > 1 and sound_detected:
            index = detected_index[-1]
            cut_out = audio_chunks[index:]
            if self.has_sound(cut_out):
                self.transcribing = True
                cut_out = self.trim_data(cut_out)
                self.data_to_queue(cut_out)
            

    def open_mic(self, buffer_size = 1024):
        pya = pyaudio.PyAudio()
        self.p = pya
        self.stream = pya.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=buffer_size)

        t = threading.Thread(target=self.mic_stream, args=(self.stream ,self.p, ))
        t.start()
        t.join()

    def cut_audio_chunks(self, detected_index, audio_chunks):
        start_index = detected_index[-2]
        last_index = detected_index[-1]
        audio_data = audio_chunks[start_index: last_index]
        audio_data = self.trim_data(audio_data)
        if len(audio_data) < 10:
            return
        self.data_to_queue(audio_data)

    def has_sound(self, audio_chunks):
        for chunk in audio_chunks:
            values = [int.from_bytes(chunk[i:i+2], 'little', signed=True) for i in range(0, len(chunk), 2)]

            if max(values) > 700:
                return True

        return False
    
    def trim_data(self, audio_chunks):
        # Trim starting
        counter = 0
        for chunk in audio_chunks:
            values = [int.from_bytes(chunk[i:i+2], 'little', signed=True) for i in range(0, len(chunk), 2)]

            if max(values) < 400:
                counter += 1
                audio_chunks.remove(chunk)
            
            else:
                break

        for chunk in reversed(audio_chunks):
            values = [int.from_bytes(chunk[i:i+2], 'little', signed=True) for i in range(0, len(chunk), 2)]

            if max(values) < 400:
                counter += 1
                audio_chunks.remove(chunk)

            else:
                break
        
        print(f"removed useless chunks: {counter}x!")
        print(f"used chunks: {len(audio_chunks)}")
        return audio_chunks