import speech_recognition as sr
import openai
from pyAudioAnalysis import audioSegmentation
from pydub import AudioSegment
import numpy as np
import redis
import hashlib

class SpeechRecognitionManager:
    def __init__(self, openai_api_key, redis_host, redis_port):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.user_signatures = {}
        self.openai_api_key = openai_api_key
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.audio = None
    def recognize_speech(self, provider):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            self.audio = self.recognizer.listen(source)
            
            if provider == "whisper":
                return self.recognize_with_whisper(self.audio)
            elif provider == "sphinx":
                return self.recognize_with_sphinx(self.audio)
            elif provider == "google":
                return self.recognize_with_google(self.audio)
            else:
                raise ValueError("Invalid provider")
    
    def recognize_with_whisper(self, audio):
        return self.recognizer.recognize_whisper_api(audio)
    
    def recognize_with_sphinx(self, audio):
        return self.recognizer.recognize_sphinx(audio)
    
    def recognize_with_google(self, audio):
        return self.recognizer.recognize_google(audio)
    
    def identify_speaker(self, audio):
        signal = np.frombuffer(audio.raw_data, dtype=np.int16)
        segments, _ = audioSegmentation.silence_removal(signal, audio.sample_rate, st_win=0.05, st_step=0.05)
        user_id = None
        is_new_user = False
        if segments:
           audio_segment = AudioSegment(
             data=audio.raw_data,
             sample_width=audio.sample_width,
             frame_rate=audio.sample_rate,
             channels=audio.channel_count
           )[segments[0][0]*1000:segments[0][1]*1000]
           user_id = self.get_user_id(audio_segment)
        if user_id is None:
            user_id = self.user_id_counter
            self.user_id_counter += 1
            is_new_user = True
        return user_id, is_new_user
    
    def get_user_id(self, audio_segment):
        # Perform diarization to get segments
        segments = self.perform_diarization(audio_segment)
        
        # Generate a signature from segments and hash it
        signature = np.concatenate(segments)
        signature_hash = hashlib.sha256(signature).hexdigest()
        
        # Check if the signature exists in Redis
        for user_id in self.redis_client.keys("user_signature:*"):
            if self.redis_client.get(user_id) == signature_hash:
                return user_id.decode("utf-8").split(":")[1]
        
        return None
    
    def generate_user_signature(self, audio_segment):
        # Perform diarization to get segments
        segments = self.perform_diarization(audio_segment)
        
        # Combine segments to create a signature
        signature = np.concatenate(segments)
        signature_hash = hashlib.sha256(signature).hexdigest()
        
        return signature_hash
    
    def save_user_signature(self, user_id, signature):
        self.redis_client.set(f"user_signature:{user_id}", signature)
    
    def recognize_user(self, audio):
        text = self.recognize_speech("sphinx")
        user_id, is_new_user = self.identify_speaker(self.audio)
        if is_new_user:
            self.user_signatures[user_id] = text
        return user_id, text
    
    def process_audio(self, provider):
        while True:
            audio = self.recognize_speech(provider)
            user_id, text = self.recognize_user(audio)
            print(f"User {user_id} said: {text}")
            
            # Process voice interactions and commands
            self.voice_interactions(user_id, text)
    
    def voice_interactions(self, user_id, text):
        if "write a blog" in text.lower():
            self.write_blog_command(user_id)
        elif "play music" in text.lower():
            self.play_music_command(user_id)
        # Add more voice interactions here
    
    def write_blog_command(self, user_id):
        user_signature = self.user_signatures.get(user_id, "Unknown")
        print(f"User {user_id} with signature '{user_signature}' wants to write a blog.")
        # Add your logic to generate the content of the blog
    
    def play_music_command(self, user_id):
        user_signature = self.user_signatures.get(user_id, "Unknown")
        print(f"User {user_id} with signature '{user_signature}' wants to play music.")
        # Add your logic to play music
    
    def perform_diarization(self, audio_segment):
        # Perform diarization using pyAudioAnalysis
        segments, _ = audioSegmentation.speaker_diarization(audio_segment.raw_data, audio_segment.frame_rate, num_of_speakers=2)
        return segments
    def get_command(self,name):
        # Get command voice command from user to talk with the assistant he will to pronunce the name of assistant
        with self.microphone as source:
             self.recognizer.adjust_for_ambient_noise(source)
        while True:
           transcript = self.recognize_speech("whisper")
           if name in transcript.lower():
               command = transcript.lower().split(name)[1]
               print (f"User said: {command}");
               return command
          
# Utilisation de la classe SpeechRecognitionManager
openai_api_key = "YOUR_OPENAI_API_KEY"
redis_host = "localhost"
redis_port = 6379
manager = SpeechRecognitionManager(openai_api_key, redis_host, redis_port)
manager.get_command("jarvis")
