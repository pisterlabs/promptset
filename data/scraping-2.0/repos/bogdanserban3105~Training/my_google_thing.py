import openai
import time
import io
import requests
import json
from google.oauth2 import service_account
from google.cloud import speech_v1
from google.cloud import speech
import pyttsx3

# Initialize OpenAI API
openai.api_key = "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"

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the Google Cloud Speech-to-Text API client
credentials = service_account.Credentials.from_service_account_file('/workspaces/114236989/talent-trainers-idm-grup1-d017d89893ea.json')
client = speech_v1.SpeechClient(credentials=credentials)


def transcribe_audio_to_text(filename):
    with io.open(filename, "rb") as f:
        content = f.read()

    audio = speech_v1.RecognitionAudio(content=content)
    config = speech_v1.RecognitionConfig(
        encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    text = ""
    for result in response.results:
        text += result.alternatives[0].transcript

    return text.strip()


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response["choices"][0]["text"]


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    while True:
        # Wait for user to say "Genius"
        print("Say 'Genius' to start recording your question")
        response = requests.get('http://localhost:5000/wait_for_keyword?keyword=genius')
        if response.status_code == 200:
            # Record audio
            filename = "input.wav"
            print("Say your question")
            response = requests.get('http://localhost:5000/record/start?filename=' + filename)
        if response.status_code == 200:
            # Transcribe audio to text
            print("Transcribing audio...")
            text = transcribe_audio_to_text(filename)
            print("You said: ", text)
            # Generate response from OpenAI API
            prompt = "I am a genius. Ask me anything."
            prompt += text
            print("Generating response...")
            response = generate_response(prompt)
            print("Response: ", response)

            # Speak response using text-to-speech engine
            speak_text(response)
            time.sleep(2)
        else:
            print("Error waiting for keyword 'Genius'. Status code: ", response.status_code)


if __name__ == "__main__":
    main()