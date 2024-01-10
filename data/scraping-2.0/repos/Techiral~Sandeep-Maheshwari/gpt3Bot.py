import openai
import pyttsx3
import speech_recognition as sr
from api_key import API_KEY


openai.api_key = API_KEY

engine = pyttsx3.init()

r = sr.Recognizer()
mic = sr.Microphone(device_index=1)


conversation = '''Sandeep Maheshwari is an Indian motivational speaker, entrepreneur, and YouTuber. He was born in Delhi in 1980, started his career as a model, and later became a professional photographer. He achieved awards such as Creative Entrepreneur of the Year and Star Youth Achiever Award. His net worth is $4 million and he owns two cars. He enjoys ice cream, and his favorite actors are Amitabh Bachchan and Sridevi. He is 42 years old and stands at 5 feet 9 inches.'''
user_name = "Audience"
bot_name = "Sandeep Maheshwari"

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

    prompt = user_name + ": " + user_input + "\n" + bot_name+ ": "

    conversation += prompt  # allows for context

    # fetch response from open AI api
    response = openai.Completion.create(engine='text-davinci-001', prompt=conversation, max_tokens=500)
    response_str = response["choices"][0]["text"].replace("\n", "")
    response_str = response_str.split(user_name + ": ", 1)[0].split(bot_name + ": ", 1)[0]

    conversation += response_str + "\n"
    print(response_str)

    engine.say(response_str)
    engine.runAndWait()
