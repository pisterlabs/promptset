import speech_recognition as sr
import pyaudio
from pydub import AudioSegment
from io import BytesIO
from kivy.core.audio import SoundLoader
import requests
import uuid
import threading as th
from enum import IntEnum
from collections import deque
import numpy as np
import json
import os
import time as t
import openai
from src.Gwen.AISystem.Networks import NonLSTMKeywordAudioModel, LSTMKeywordAudioModel
import torch as torch
from src.utils import get_mel_image_from_float_normalized, normalize_mfcc, get_json_variables

class AudioController(object):
    class SoundClip(IntEnum):
        ENGAGED = 0
        OPENING_NETFLIX = 1
        OPENING_SPOTIFY = 2
        OPENING_YOUTUBE = 3
        CLOSING_APPLICATIONS = 4
        
    
    class State(object):
        class Mode(IntEnum):
            LISTENING = 0
            STREAMING = 1
            TEMP = 2
            
            DATA_COLLETION = 5
            TRAP = 10
            
        def __init__(self, ):
            self.mode = AudioController.State.Mode.LISTENING
        
        def transition(self, set=-1):
            if not set == -1:
                self.mode = AudioController.State.Mode(set)
                return self.mode != AudioController.State.Mode.TRAP
            self.mode = AudioController.State.Mode((int(self.mode) + 1) % 3)
            
        def __call__(self,):
            return self.mode
    
    def __init__(self, GwenInstance):
        '''Basic Audio Control'''
        print('Initializing Audio Controller...')
        self.GwenInstance = GwenInstance
        self.mic = pyaudio.PyAudio()
        self.r = sr.Recognizer()
        self._audio_buffer = None
        self.sound_loader = None
        
        self.state = AudioController.State()
        self._currently_speaking = False
        
        self._config = get_json_variables(os.path.join(os.getcwd(), 'data', 'Gwen', 'Audio', 'AudioControllerConfig.json'), ["model_path","data_output_path","n_mfcc","max_len", "tts_header", "tts_body"])
        
        openai.api_key = os.environ.get('OPENAI_API_KEY')        
        self.n_mfcc = self._config["n_mfcc"]
        self.max_len = self._config["max_len"]
        
        self.__audio_buffer_lock = th.Lock()
        self.reset_audio_buffer()
                
        # -- Neural Data -- #
        self._current_stream_img = self.buffer_to_img()    
        self._prediction_model = LSTMKeywordAudioModel.Load_Model(os.path.join(os.getcwd(), self._config["model_path"]))
        self._data_output_path = os.path.join(os.getcwd(), self._config["data_output_path"])
        
        # -- Azure TTS -- #
        self._tts_header = json.loads(self._config["tts_header"].replace("<SubscriptionKey>", os.environ.get("AZURE_API_KEY")).replace("<ResourceName>", os.environ.get("AZURE_RESOURCE_NAME")))
        self._tts_body = self._config["tts_body"]
        self._tts_url = 'https://eastus.tts.speech.microsoft.com/cognitiveservices/v1'

        
        self._stream = None
        self.listen_for_keyword()
        
        
    def buffer_to_img(self):
        with self.__audio_buffer_lock:
           return normalize_mfcc(np.vstack(np.array(self._audio_buffer).copy()))
       
    def listen_for_keyword(self):
        self._stream = self.record_audio()
       
    def record_audio(self) -> pyaudio.Stream:
        """
        Streams Microphone Audio and stores it in a buffer as a Mel Image.
        """
        audio_format = pyaudio.paInt16
        num_channels = 1
        sample_rate = 48000
        duration = 0.25        
        frame_length = int(sample_rate*duration)

        def __audio_stream_callback__(in_data, a, b, c):
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            audio_data_np = np.frombuffer(audio_data.tobytes(), dtype=np.int16).astype(np.float32)
            audio_data_np /= np.iinfo(np.int16).max # Norm the stream data
            with self.__audio_buffer_lock: # Using this buffer lock ensures that we do not read from a different thread
                self._audio_buffer.popleft()
                mel_img = get_mel_image_from_float_normalized(audio_data_np, sound_rate=sample_rate)
                self._audio_buffer.append(mel_img)
            return (in_data, pyaudio.paContinue)

        # Open stream on our microphone
        stream = self.mic.open(format=audio_format,channels=num_channels, rate=sample_rate, input=True, frames_per_buffer=frame_length, stream_callback=__audio_stream_callback__)
        stream.start_stream()
        return stream
    
    # --- Application API --- #

    def run(self, data, is_main_context=False) -> None:
        if self._currently_speaking:
            return
        
        if self.state() == AudioController.State.Mode.LISTENING:
            # Run Prediction on the Current Audio Stream 
            prediction = self.get_prediction()
            if prediction: # Stop the stream and Transition to Command Parsing
                print('Keyword detected...')
                if self._currently_speaking:
                    self.sound_loader.stop()
                    
                with self.__audio_buffer_lock: # Using this buffer lock ensures that we do not read from a different thread
                    self._stream.stop_stream()
                    self._stream.close()
                    self.mic.terminate()
                    self.state.transition()
                    self._prediction_model.reset_hidden_state()
                
        elif self.state() == AudioController.State.Mode.STREAMING:
            # Use Microphone Audio to Predict Command-String
            source = sr.Microphone()
            tmp_path = os.path.join(os.getcwd(), "tmp", "data", "sound", "temp_audio.wav")
            with source as source:
                    self.r.adjust_for_ambient_noise(source=source, duration=1.5)                                                             
                    try:
                        print('Engaged Gwen System...')
                        temp_audio = self.r.listen(source) 
                        with open(tmp_path, 'wb') as tmp: # Create Temp Audio File for Whisper API
                            tmp.write(temp_audio.get_wav_data())
                        # Open and make API Call
                        with open(tmp_path, 'rb') as audio_file:
                            response = openai.Audio.transcribe("whisper-1", audio_file) ["text"]
                        self.state.transition(AudioController.State.Mode.TEMP)
                        print('Processing Response...')
                        self.GwenInstance.execute_command(response)
                    except sr.UnknownValueError as e:
                        print("Could not understand audio")
                        self.state.transition(AudioController.State.Mode.LISTENING)
                    except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
                        self.state.transition(AudioController.State.Mode.LISTENING)
                    finally:
                        os.remove(tmp_path)
        elif self.state() == AudioController.State.Mode.TEMP:
            self.reset_audio_buffer()
            self.listen_for_keyword()
            self.state.transition(AudioController.State.Mode.LISTENING) # Asssuming Context took care of the rest.
        
        elif self.state() == AudioController.State.Mode.DATA_COLLETION:
            if not is_main_context: # TODO: Implement Viable Data Collection solution for Contexts.
                self.state.transition(AudioController.State.Mode.LISTENING) # Asssuming Context took care of the rest.
        
        elif self.state() == AudioController.State.Mode.TRAP:
            '''Bad state :( hope I don't end up here '''
        
        t.sleep(0.05)
        
    def transition_mode(self, mode=-1):
        self.state.transition(mode)
    
    def reset_audio_buffer(self):
        self._audio_buffer = deque() 
        with self.__audio_buffer_lock :
            for _ in range(4):
                self._audio_buffer.append(np.zeros(shape=(self.n_mfcc, self.max_len), dtype=np.float32)) 
            
    def get_prediction(self,):
        img = self.buffer_to_img()
        while len(img.shape) < 5:
            img = np.expand_dims(img, axis=0)
        try:
            return self._prediction_model.predict(torch.tensor(img, device=self._prediction_model.device))
        except Exception as e:
            print(e)
            return False # TODO: Add Logging of errors.
        
        
    def exec(self, data):
        if data['func'] == 'DataCollection':
            self.state.transition(AudioController.State.Mode.DATA_COLLETION)
            self.collect_data_sequence(num_samples=data['num_samples'], path=self._data_output_path)
        elif data['func'] == '':
            pass # Add more Cases eventually.
    
    def quit(self,): # Ignore call to quit.
        pass
    
    # --- Data Collection API --- #
    def collect_data_sequence(self, num_samples, path, sample_episode=100, time_low=5, time_high=20):
        print('While collecting data, you will be prompted to provide audio. When prompted, make sure to respond promptly.\nThis will ensure proper data collection.')
        self._stream.stop_stream()
        self._stream.close()
        self.mic.terminate()
        source = sr.Microphone()
        for _ in range(num_samples):
            with source as source:  
                index = len(os.listdir(path))
                record_time = np.random.randint(low=time_low, high=time_high)
                try:  
                    print('Recording for ' + str(record_time) + ' seconds...')  
                    t.sleep(1.0)
                    print('--- START ---')
                    print('...Collecting...')
                        
                    audio = self.r.record(source=source, duration=record_time)
                    
                    print('--- Finished Recording Sample ---')
                    with open(os.path.join(path, f'{index}.wav'), 'wb') as f:
                        f.write(audio.get_wav_data())
                    print('--- Finished Saving Sample (\..Enter to Continue../) ---')
                    input()
                        # t.sleep(1.0)
                except sr.UnknownValueError:
                        print("Could not understand audio")
                except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
            
        self.state.transition(AudioController.State.Mode.LISTENING) # Transition to listening mode
    
    # --- Audio Playback API --- #
    
    def speech_output(self, speech: str):
        def thr_speak():
            try:
                path = self.convert_tts(speech)
                self.play_audio(path)
                os.remove(path) # Cleanup
            except Exception as e:
                print(e)
        s_t = th.Thread(target=thr_speak)
        s_t.start()
        s_t.join()            
    
    def play_clip(self, sound_clip: 'AudioController.SoundClip'):
        pass
    
    def play_audio(self, path: str):
        def _on_play_end(args):
            self._currently_speaking = False
        self.sound_loader = SoundLoader.load(path)
        self.sound_loader.bind(on_stop= _on_play_end)
        self.sound_loader.play()
        self._currently_speaking = True
        
    def convert_tts(self, speech: str):
        path = os.path.join(os.getcwd(), "tmp", "data", "sound", uuid.uuid4().hex + ".wav") # Temp File Path
        response = requests.post(self._tts_url, headers=self._tts_header, data=self._tts_body.replace("$BODY_SPEECH$", speech).encode('utf-8'))
        
        if response.status_code == 200:
            # Convert to WAV and store at temp path
            audio = AudioSegment.from_file(BytesIO(response.content), format="wav")
            audio.export(path, format="wav")
            return path
        else: # Request Failed
            raise Exception(f"Failed to convert Speech to WAV. Status Code: {response.status_code} Reason:{response.text}")