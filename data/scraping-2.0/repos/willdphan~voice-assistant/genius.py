import openai
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import time
import os

# Initialize OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize the text to speech engine stored in engine var
engine = pyttsx3.init()

# speech recognition lib from audio as input and transcribes it to text
# takes in arg 'filename' which specifies the audio file we want
# to transcribe
def transcribe_audio_to_test(filename):
    # create instance of recognizer class from sr module
    recogizer = sr.Recognizer()
    # with statement to open audio file using audio file
    # class from sr module
    with sr.AudioFile(filename) as source:
        # record the audio using the record method of the recognizer object
        audio = recogizer.record(source) 
    try:
        # transcribes the audio into text using the recognize_google method
        return recogizer.recognize_google(audio)
        # if an err occures, and message will be displayed
    except:
        print("skipping unkown error")

# generates response from gpt3 api
# takes single argument prompt - which represents input text
# as starting point for generating a repsonse using gpt3 api
def generate_response(prompt):
    # pass several arguments to specify parameters of response
    response= openai.completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0.5,
    )
    # return gen response from gpt3 api
    return response ["Choices"][0]["text"]

# takes text arg that converts it to speech using pyttsx3 lib
def speak_text(text):
    # specifies the text to be spoken
    engine.say(text)
    # plays the speech
    engine.runAndWait()

# shows how we want python to run the script
def main():
    # runs continuously until program is stopped
    while True:
        # Wait for user say "genius"
        print("Say 'Genius' to start recording your question")
        # sr microphone class to access mic and audio
        with sr.Microphone() as source:
            # creates the instance of sr recognizer class
            recognizer = sr.Recognizer()
            # records the audio using the listen method of recognizer object
            audio = recognizer.listen(source)
            # transcibes recorded audio to text using the recognize google method
            try:
                # next two lines check if the transcribed text is genius and
                # the lower method converts it to lower case to make it 
                # case insensitive
                transcription = recognizer.recognize_google(audio)
                # if it is "genius" then record more audio
                if transcription.lower() == "genius":
                    # record audio and save it to file input.wav
                    filename ="input.wav"
                    # used to display message to say the question
                    print("Say your question")
                    with sr.Microphone() as source:
                        recognizer=sr.recognize()
                        source.pause_threshold=1
                        audio=recognizer.listen(source, phrase_time_limit = None, timeout = None)
                        with open(filename,"wb") as f:
                            f.write(audio.get_wav_data())
                        
                    # transcribes audio to test 
                    text = transcribe_audio_to_test(filename)
                    # if the transcription was successful, the text var will contain
                    # the transcribed text and print it
                    if text:
                        print(f"you said {text}")
                        
                        # Generates the response and prints the response
                        response = generate_response(text)
                        print(f"chat gpt 3 say {response}")
                            
                        # Reads the response using text to speech
                        speak_text(response)

            # handles error with printed response
            except Exception as e:
                print("An error ocurred : {}".format(e))

            # code that runs the main func
            if __name__=="__main__":
                main()

