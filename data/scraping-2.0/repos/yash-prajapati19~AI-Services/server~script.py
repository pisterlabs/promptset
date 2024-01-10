from datetime import datetime
import speech_recognition as sr
import pyttsx3
import keyboard
import openai
import os

openai.api_key = "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"

#from dotenv import load_dotenv
#load_dotenv()
expected = ["/n/nDoctor","/n/nJob-requirement","/n/nLawyer","/n/nHouse-rental"]
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
activationWord = 'circuit'

def speak(text, rate = 120):
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

def parseCommand():
    listener = sr.Recognizer()
    print('Listening for a command')

    with sr.Microphone() as source:
        listener.pause_threshold = 2
        input_speech = listener.listen(source)

    try:
        print('Recognition speech')
        query = listener.recognize_google(input_speech, language='en_gb')
        print(f'The input speech was: {query}')
    except Exception as exception:
        print('I did not quite catch that')
        speak('I did not quite catch that')
        print(exception)
        return 'None'

    return query

def generate_response(user_input):

  model_engine = "text-davinci-002"
  prompt = '"'+user_input+'". I want you to tell me that which kind of service a user need from the sentence mention above and answer me in one word from this options (Doctor, Lawyer, Job-requirement, House-rental) and if you could not find answer from 4 options then return "unrecog"'
  max_tokens = 100
  temperature = 0.6

  completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature
  )

  ai_response = completion.choices[0].text

  return ai_response

def generate_response2(user_input):

  model_engine = "text-davinci-002"
  prompt = '"'+user_input+'". From the above sentence  extract the city name and answer in one word, if you are not able to find city name then return "nocity".'
  max_tokens = 100
  temperature = 0.6

  completion = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=max_tokens,
    temperature=temperature
  )

  ai_response = completion.choices[0].text

  return ai_response

if __name__=='__main__':
    speak('All systems nominal.')
    task = True
    while task:
        query = parseCommand().lower().split()

        if query[0] == activationWord:
            query.pop(0)

            userinput = ' '.join(query)
            print(f'User input: {userinput}')
            output = generate_response(userinput)
            print(output)
            kaam = True
            while kaam:
                if output in expected:
                    print(f'User input: In which city you will prefer this service')
                    userinput2 = parseCommand().lower().split()
                    print(f'User input: {userinput2}')
                    output2 = generate_response2(output)
                    print(output2)
                    kaam = False
            #    else:
            #        print("Mention city name to get accurate results")
        task = False
    else:
        print("Sorry we couldn't recognise your problem to get accurate results.")
