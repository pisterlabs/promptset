import openai
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
speech_key, service_region = os.getenv("AZURE_SPEECH_KEY"), os.getenv("AZURE_SERVICE_REGION")
text_analytics_key, text_analytics_endpoint = os.getenv("AZURE_TEXT_ANALYTICS_KEY"), os.getenv("AZURE_TEXT_ANALYTICS_ENDPOINT")

# Initialize Azure Speech Services
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

# Initialize Azure Text Analytics
text_analytics_client = TextAnalyticsClient(endpoint=text_analytics_endpoint, credential=AzureKeyCredential(text_analytics_key))

def detect_language(text):
    response = text_analytics_client.detect_language(documents=[{"id": "1", "text": text}])
    language = response[0].primary_language.iso6391_name
    return language

def get_voice_input():
    print("Listening...")

    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        detected_language = detect_language(result.text)
        speech_config.speech_recognition_language = detected_language
        print(f"Detected language: {detected_language}")
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized. Try again.")
        return None
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech recognition was canceled. Try again.")
        return None

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

def speak(text):
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized successfully.")
    elif result.reason == speechsdk.ResultReason.Canceled:
        print("Speech synthesis was canceled.")
        return None

if __name__ == "__main__":
    conversation = {}
    print("Welcome to the GPT-3.5 voice chatbot!")
    while True:
        user_input = get_voice_input()
        if user_input:
            print(f"User: {user_input}")
            prompt = f"The user said: {user_input}. How should the chatbot respond?"
            chatbot_response = generate_response(prompt)
            print(f"Chatbot: {chatbot_response}")
            speak(chatbot_response)

            # Store the user input and chatbot response in the conversation dictionary
            conversation[user_input] = chatbot_response
        else:
            print("Please try again.")