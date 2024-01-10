import os

from openai import OpenAI, OpenAIError
import base64 as b64
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sentiment_analyzer = SentimentIntensityAnalyzer()


class ProcessingException(Exception):
    def __init__(self, message="OpenAI exception occurred"):
        self.message = message
        super().__init__(self.message)


class Processing:
    def __init__(self, transcription: str, summary: str, sentiment: Dict[str, float]):
        self.transcription = transcription
        self.summary = summary
        self.sentiment = sentiment


# raises ProcessingException
def process_audio(base64) -> Processing:
    transcription = transcribe(base64)
    summary = sum_up(transcription)
    sentiment = run_sentiment_analysis(transcription)
    result = Processing(transcription, summary, sentiment)
    return result


def transcribe(base64) -> str:
    key = os.environ.get('OPENAI_API_KEY')
    client = OpenAI(api_key=key)
    mp3_data = b64.b64decode(base64)
    audio_file_path = "audio.mp3"
    with open(audio_file_path, "wb") as mp3_file:
        mp3_file.write(mp3_data)
    with open(audio_file_path, "rb") as audio_file:
        try:
            transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file).text
        except OpenAIError:
            os.remove(audio_file_path)
            raise ProcessingException("Error occurred while trying to transcribe audio file.")
    os.remove(audio_file_path)
    return transcript


def run_sentiment_analysis(text: str) -> Dict[str, float]:
    return sentiment_analyzer.polarity_scores(text)


def sum_up(text: str) -> str:
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in summarizing text."},
                {"role": "user", "content": f"""You will receive a transcription of an audio file (text). Your task is to create a summary of 
                    this text. You have to be concise and use english no matter what the original language is.
                    Answer immediately without any additional introductions or explanation, just a summary. 
                    This is the input: {text}"""}
            ]
        )
    except OpenAIError:
        raise ProcessingException("Error occurred while trying to create a summary of text.")
    return response.choices[0].message.content
