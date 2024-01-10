import speech_recognition as sr
from time import sleep
import openai
import os
import subprocess
import firebase_admin
from datetime import datetime
from firebase_admin import credentials
from firebase_admin import firestore

# Use a service account.
cred = credentials.Certificate('path/to/serviceAccount.json')

app = firebase_admin.initialize_app(cred)

db = firestore.client()

client = openai.OpenAI(api_key='')
messages = [{"role": "system", "content": "You are a gardening assistant who will answer questions regarding gardening only with a max of 3 sentences. Say uwu at the end of every response. If asked any unrelated question, respond with uncertainty"}]

r = sr.Recognizer()
mic = sr.Microphone()

print("starting")

def listen_for_start_word(source):
    print("Listening for hey seed...")
    while True:
        audio = r.listen(source)
        print(audio)
        try:
            words = r.recognize_google(audio)
            print(words)
            if "hey seed" in words.lower():
                print("Wake word detected.")

                listen_and_respond(source)
                break
        except sr.UnknownValueError:
            pass

def listen_and_respond(source):
    print("Listening for question ...")

    while True:
        audio = r.listen(source)
        try:
            words = r.recognize_google(audio)
            print(f"You said: {words}")
            if not words:
                continue
        
            messages.append({"role": "user", "content": words})
            print('waiting for gpt response')
            subprocess.run(['espeak', '-ven+f3', '-s130', 'Of course! I would love to answer you. Give me a second uwu', '2>/dev/null'])

            # saving user command to db
            try: 
                data = {'message': words, 'time': datetime.now(), 'type': 'user'}

                db.collection('gpt').document(words).set(data)
                print('saved user response to db')
            except Exception as e:
                print('error', e)

            # Use the correct syntax to access the response
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            reply = response.choices[0].message.content

            print('speaking...')
            subprocess.run(['espeak', '-ven+f3', '-s130', reply, '2>/dev/null'])

            messages.append({"role": "assistant", "content": reply})

            # saving gpt response to db
            try: 
                data = {'message': reply, 'time': datetime.now(), 'type': 'bot'}

                db.collection('gpt').document(reply).set(data)
                print('saved bot response to db')
            except Exception as e:
                print('error', e)

            print("\n" + reply + "\n")

            listen_for_start_word(source)


        except sr.UnknownValueError:
            sleep(2)
            print("Silence found, shutting up, listening...")
            listen_for_start_word(source)
            break


with sr.Microphone() as source:
    listen_for_start_word(source)
