import openai
import speech_recognition as sr
import pyttsx3
import time

#Inicializa api OpenAI
openai.api_key = "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"
#SpeechToText
engine=pyttsx3.init()

def transcribe_audio_to_test(filename):
    recogizer=sr.Recognizer()
    with sr.AudioFile(filename)as source:
        audio=recogizer.record(source) 
    try:
        return recogizer.recognize_google(audio, language="es")
    except:
        print("No se que ha pasao")

def generate_response(prompt):
    response= openai.Completion.create(
        engine="text-curie-001",
        prompt=prompt,
        max_tokens=2049,
        n=1,
        stop=None,
        temperature=0.5,
    )
    return response ["choices"][0]["text"]

# Set Spanish voice for text-to-speech engine
voices = engine.getProperty('voices')
spanish_voice = None
for voice in voices:
    if "spanish" in voice.languages:
        spanish_voice = voice.id
if spanish_voice is not None:
    engine.setProperty('voice', spanish_voice)

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def main():
    while True:
        #Waith for user say "genius"
        print("Di 'Hola' para empezar a grabar")
        with sr.Microphone() as source:
            recognizer=sr.Recognizer()
            audio=recognizer.listen(source)
            try:
                transcription = recognizer.recognize_google(audio, language="es")
                if transcription.lower()=="hola":
                    #record audio
                    filename ="input.wav"
                    print("Dime que quieres mozo")
                    with sr.Microphone() as source:
                        recognizer=sr.Recognizer()
                        source.pause_threshold=1
                        audio=recognizer.listen(source,phrase_time_limit=None,timeout=None)
                        with open(filename,"wb")as f:
                            f.write(audio.get_wav_data())
                    #transcript audio to test 
                    text=transcribe_audio_to_test(filename)
                    if text:
                        print(f"Yo {text}")
                        
                        #Generate the response
                        response = generate_response(text)
                        print(f"El bot ese dice: {response}")
                            
                        #read resopnse using GPT3
                        speak_text(response)
            except Exception as e:
                
                print("Ahhhhhh erroor : {}".format(e))
if __name__=="__main__":
    main()