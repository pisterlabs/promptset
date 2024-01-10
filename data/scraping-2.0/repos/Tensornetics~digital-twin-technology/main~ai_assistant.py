import openai_api
import nvidia.nemo.nemo_asr
import nvidia.nemo.nemo_nlp
import nvidia.nemo.nemo_tts
import riva.riva_asr

class AIAssistant:
    def __init__(self):
        # Initialize the OpenAI API client
        self.openai = openai_api.OpenAI()
        
        # Initialize the NVIDIA NEMO ASR, NLP, and TTS clients
        self.nemo_asr = nvidia.nemo.nemo_asr.NemoASR()
        self.nemo_nlp = nvidia.nemo.nemo_nlp.NemoNLP()
        self.nemo_tts = nvidia.nemo.nemo_tts.NemoTTS()
        
        # Initialize the Riva ASR client
        self.riva_asr = riva.riva_asr.RivaAS
