import openai
import threading
from transformers import pipeline
import numpy as np
from scipy.io.wavfile import read

class Audio_Analysis:

    def __init__(self, audio_path):
        #model inits
        self.sentiment_analysis_pipeline = pipeline('sentiment-analysis', model = 'distilbert-base-uncased-finetuned-sst-2-english')

        #whisper inits
        self.openaikey = ""

        #data inits
        self.audio_path = audio_path
        self.transcription_array = []
        self.transcription = ""
        self.sentiment_analysis = []
        self.emotion_detection = []
        self.keywords = ()
        self.energy_level = "Low"
    
    def transcribe_audio(self):
        openai.api_key = self.openaikey
        with open(self.audio_path, "rb") as audio_file:
            transcription = openai.Audio.transcribe("whisper-1", audio_file)
            self.transcription = transcription["text"]
            self.transcription_array = transcription["text"].split('.')

    def run_energy_detection(self):
        with open(self.audio_path, "rb") as audio_file:
            fs, amplitude = read(audio_file)

            avg_amplitude = np.mean(np.abs(amplitude))
            if avg_amplitude > 500:
                self.energy_level = "High"
    
    def run_sentiment_analysis(self):
        sentiment_analysis_results = self.sentiment_analysis_pipeline(self.transcription_array)

        self.sentiment_analysis = self.convert_analysis_result_to_array(sentiment_analysis_results)

    def convert_analysis_result_to_array(self, data):
        formatted_data = []
        for sentence_result in data:
            if sentence_result['label'] not in formatted_data:
                formatted_data.append(sentence_result['label'])
        
        return formatted_data


    def print_audio_results(self):
        f = open('detection_results.txt', 'a')

        f.write("\nTranscription: " + str(self.transcription))
        f.write("\nSentiment Analysis: " + str(self.sentiment_analysis))
        f.write("\nEmotion Detection: " + str(self.emotion_detection))
        f.write("\nKeywords: "   + str(self.keywords))
        f.write("\nEnergy Level: " + str(self.energy_level))

        f.close()

    def start_analysis(self, openaikey):
        try:
            self.openaikey = openaikey
            self.transcribe_audio()
            
            sentiment_analysis_thread = threading.Thread(target=self.run_sentiment_analysis)
            energy_detection_thread = threading.Thread(target=self.run_energy_detection)

            sentiment_analysis_thread.start()
            energy_detection_thread.start()

            sentiment_analysis_thread.join()
            energy_detection_thread.join()
        except Exception as e:
            print("Audio Exception:" + str(e))
