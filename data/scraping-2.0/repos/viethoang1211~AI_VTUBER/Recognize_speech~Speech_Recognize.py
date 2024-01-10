import speech_recognition as sr
import openai
import threading
import logging
openai.api_key=""
semaphore = threading.Semaphore(1)
def listen_and_respond():
    # Create a recognizer object
    r = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something...")
        
        while True:
            # Listen for audio input
            print("waiting for input...")
            semaphore.acquire()
            audio = r.listen(source,phrase_time_limit=15)
            
            print("recieved audio:")
            # try:
                # Use the recognizer to convert speech to text
                
            text = r.recognize_sphinx(audio)
            if len(text) <= 15:
                semaphore.release()
                continue
            else:
                pass
            
            with open("temp.wav", "wb") as f:
                f.write(audio.get_wav_data())
            
            print("You said: " + text)
            print("I heard that.")
            
            audio_file= open("temp.wav", "rb")
            trans = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file,
                temperature=0.1,
                language="en"
            )

            if len(trans['text']) <=15:
                semaphore.release()
                continue
            else:
                pass
            
            print("You: " + trans['text'])
            semaphore.release()
            # except sr.UnknownValueError:
            #     print("Sorry, I couldn't understand your speech.")
                
            # except sr.RequestError as e:
            #     print("An error occurred while processing your request: " + str(e))

# Call the function to start listening and responding
listen_and_respond()
