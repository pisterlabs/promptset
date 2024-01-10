import openai
import pyttsx3
import speech_recognition as sr
from api_key import API_KEY
from googletrans import Translator
# import keyboard

openai.api_key = API_KEY

engine = pyttsx3.init()

r = sr.Recognizer()
mic = sr.Microphone()

translator = Translator()

conversation = ""
user_name = "You"
bot_name = "Jarvis"

while True:

    with mic as source:
        print("\nlistening...")
        r.adjust_for_ambient_noise(source, duration=0.2)

        
        try:
            audio = r.listen(source, timeout=0.2)
            print("Audio captured.")
        except sr.WaitTimeoutError:
            print("Timeout Error: No speech detected.")
            continue
        except Exception as e:
            print("Error:", e)
            continue

    print("no longer listening.\n")


    try:
        user_input = r.recognize_google(audio)
        print("Recognized:", user_input)

    except:
        continue
    
    
    lang = translator.detect(user_input).lang

    
    if lang == 'bn':
        user_input = translator.translate(user_input, dest='en').text

    prompt = user_name + ": " + user_input + "\n" + bot_name+ ": "

    conversation += prompt  

    # fetch response from open AI api
    response = openai.Completion.create(engine='text-davinci-003', prompt=conversation, max_tokens=100)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

    conversation += response_str + "\n"

    
    if lang == 'bn':
        response_str = translator.translate(response_str, dest='bn').text

    print(response_str)

    engine.say(response_str)
    engine.runAndWait()
