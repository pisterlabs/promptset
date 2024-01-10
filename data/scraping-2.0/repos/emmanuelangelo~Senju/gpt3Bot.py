import openai
import pyttsx3
import speech_recognition as sr
from api_key import API_KEY
import pvporcupine
import time

openai.api_key = API_KEY

engine = pyttsx3.init()

voice = engine.getProperty('voices')

engine.setProperty('voice', voice[0].id)

r = sr.Recognizer()
mic = sr.Microphone(device_index=1)

conversation = ""
user_name = "Maxwell"
bot_name = "SENJU"

while True:
    with mic as source:
        print("\nlistening...")
        r.adjust_for_ambient_noise(source, duration=0.2)
        audio = r.listen(source)
    print("no longer listening.\n")

    try:
        user_input = r.recognize_google(audio)
    except:
        continue

    prompt = user_name + ": " + user_input + "\n" + bot_name + ": "

    conversation += prompt  # allows for context

    # fetch response from open AI api
    response = openai.Completion.create(
        engine='text-davinci-003', prompt=conversation, max_tokens=100)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(
        user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

    conversation += response_str + "\n"
    print(response_str)

    engine.say(response_str)
    engine.runAndWait()

# Create an instance of the Porcupine library
handle = pvporcupine.create(keywords=["porcupine"],
 model_file_path="porcupine_params.pv", sensitivity=0.5)


# Callback function that will be called when the wake word is detected
def wake_word_detected():
    print("Wake word detected! Starting AI voice assistant...")
    # Trigger AI voice assistant to start listening for commands
    # ...


# Start wake word detection
pvporcupine.start_detection(handle=handle, callback=wake_word_detected)

# Run your code here
# ...

# Stop wake word detection when you want
pvporcupine.stop_detection(handle=handle)

# Delete Porcupine instance when done
pvporcupine.delete(handle)
