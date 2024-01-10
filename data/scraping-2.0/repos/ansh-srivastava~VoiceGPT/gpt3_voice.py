import openai #that will allow us to import the openai gpt3 API
import pyttsx3 #this thing will allow us to convert text to speech
import speech_recognition as sr # to transcribe audio into the text
import time

# set your openAPI key
openai.api_key="YOUR OPENAI KEY"
# TO get your openai key visit:- "https://platform.openai.com" then follow the steps:- Personal > View API Keys > Create New key



# Initialise Text-To-Speech engine
engine = pyttsx3.init()

def transcribe_audio_to_text(filename):  # created a python function to transcribe the audio into the text using the python speech_recognition module
    # it also provide the convenient way to transcribe the speech into the text
    # And also it specify the audio file that it took only one argument at a time
    recognizer = sr.Recognizer() # This module has been performed to record the speech recognisation in the audio file
    with sr.AudioFile(filename) as source: # this module use with statement to open the audio file specified by filenaem using the audio file class name using the sr module
        audio = recognizer.record(source) # Then we record the audio using the 'record' method of the "recognizer" module
        try:
            return recognizer.recognize_google(audio) # Finally let's transcribe the recorded audio to text using the 'recognize_google' method under the recognizer object
        except:
            print("Skipping unknown error") # If an error occured during the transcription the exception would be raised and error would be printed and trying again would be printed

def generate_response(prompt): 
    # This function will be used to generate responsed using the GPT3 API  and also it specifies that it would take the single input prompt
   
    response = openai.Completion.create( # Using OpenAi 'Completion.create' method to generate the response based on the given prompt
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        n=1,
        stop=None,
        temprature=0.5,
    ) # above part specify the parameters of the prompt
    return response["choices"][0]["text"] # Here we return the generated response from the openai gpt3

def speak_text(text): # That will convert the text argument into speech using the pyttsx3 module
    engine.say(text)  # Specify the text to be spoken
    engine.runAndWait()

def main(): # main function will be used to run all those script
    while True:
        # Wait for user to say "genius"
        print("Say 'Genius' to start recording your question...") # The command prompt will run or execute only when it get the command 'Genius'
        with sr.Microphone() as source: # 'sr' microphone class to access the microphone of the computer to record audio
            recognizer = sr.Recognizer()
            audio = recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio)
                if transcription.lower() =='genuis':
                    # Record Audio
                    filename = "input.wav"
                    print("Say your question...")
                    with sr.Microphone() as source:
                        recognizer = sr.Recognizer()
                        source.pause_threshold = 1
                        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
                        with open(filename, "wb") as f:
                            f.write(audio.get_wav_data())
                    
                    # Transcribe audio to text
                    text = transcribe_audio_to_text(filename)
                    if text:
                        print(f"You said: {text}")

                        # Generate response using GPT-3
                        response = generate_response(text)
                        print(f"GPT-3 says: {response}")

                        # Read response using text-to-speech
                        speak_text(response)
            
            except Exception as e:
                print("An error occured: {}".format(e))

if __name__ == "__main__" :
    main()


