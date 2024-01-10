import os
from dotenv import load_dotenv
import openai
import pyttsx3
import speech_recognition as sr
import time
import keyboard

# Set you OpenAI API key
# Never commit your key to your repository
# openai.api_key="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"
load_dotenv()
openai.api_key = os.getenv("API_KEY")
# alternatively, you can set the API key as an environment variable
# secrets = dotenv_values(".env")

# Otherwise, you can set the API key as an environment variable
# https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety

# Initialize the text-to-speech engine
engine = pyttsx3.init()

def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()

    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)

        try:  
            return recognizer.recognize_google(
                                            audio,
                                            language="da-DK"
                                        )
        except:
            print("There was an error transcribing the audio to text")
            return None
    
def generate_response(prompt):
    response = openai.Completion.create(
      #engine="davinci",
      engine="text-davinci-003",
      prompt=prompt,
      temperature=0.9,
      max_tokens=150,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0.6,
      #stop=["\n", " Human:", " AI:"]
      stop=None
    )
    return response["choices"][0]["text"]

def speak_text(text):
    engine.say(
        text
        )
    engine.runAndWait()
    
def main():
    while True:
        # Record audio
        print("Say Genius to start recording ...")
        
        with sr.Microphone() as source:
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)

            try:
                transcript = recognizer.recognize_google(
                                                    audio,
                                                    language="da-DK"
                                                    )
                if transcript.lower() == "genius":
                    
                    # Record audio
                    filename = "input.wav"
                    print("Say your question ...")
                    
                    with sr.Microphone() as source:
                        recognizer.adjust_for_ambient_noise(source)
                        source.pause_threshold = 1
                        audio = recognizer.listen(
                                                source, 
                                                phrase_time_limit=None, 
                                                timeout=None
                                            )
                        
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())
                            print("Done recording")

                    # Transcribe audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print("Transcription: " + text)
                        
                        # Generate response using GPT-3
                        response = generate_response(text)
                        print("GPT-3 Response: {response}")
                    
                        # Read response aloud using text-to-speech
                        speak_text(response)
        
                else:
                    print("The transcript was: {}".format(transcript.lower()))
                        
                        
            except Exception as e:
                print("There was an error {}".format(e))


if __name__ == "__main__":
    main()
