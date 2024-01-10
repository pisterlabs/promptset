import speech_recognition as sr
import os
import webbrowser
import openai
from key import OPENAI_API_KEY as apikey
import datetime
import win32com.client as wincl
from newsapi import NewsApiClient

speaker = wincl.Dispatch("SAPI.SpVoice")
name = input("Enter your name: ")

def chat(query):
    openai.api_key = apikey
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": query
        }
    ],
    temperature=1,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    speaker.Speak(response['choices'][0]['message']['content'])


def ai(prompt):
    openai.api_key = apikey
    text = f"{prompt}\n\n"

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "user",
        "content": query
        }
    ],
    temperature=1,
    max_tokens=50,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    text += response['choices'][0]['message']['content']
    if not os.path.exists("AI"):
        os.mkdir("AI")

    with open(f"AI/file.txt", "w") as f:
        f.write(text)

    speaker.Speak("Done. Please check the file")

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language="en-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Sorry, I didn't get that. Please say again"


speaker.Speak(f"Hello {name}...How can I help you?")
while True:
    query = "open movies"
    sites = [["youtube", "https://www.youtube.com"], ["ChatGPT", "https://chat.openai.com/"], ["google", "https://www.google.com"],]
    for site in sites:
        if f"Open {site[0]}".lower() in query.lower():
            speaker.Speak(f"Opening {site[0]} ")
            webbrowser.open(site[1])

    if "open movies" in query.lower():
        speaker.Speak("Opening movies folder")
        musicPath = "c:/Users/kiran/Videos/Movies"
        os.startfile(musicPath)

    elif "time" in query.lower():
        time = datetime.datetime.now().strftime("%H:%M %p")
        speaker.Speak(f"The time is {time}")

    elif "Using artificial intelligence".lower() in query.lower():
        speaker.Speak("writing file")
        ai(query)

    elif 'news' in query.lower():
        speaker.Speak("Here are some top headlines")
        newsapi = NewsApiClient(api_key='ec9b9a43d66a40a3ada4b8b2e2481cce')
        top_headlines = newsapi.get_top_headlines(sources='bbc-news,times-of-india,google-news-in,the-hindu,news24')
        for i in top_headlines['articles']:
            speaker.Speak(i['title'])

    elif "Exit".lower() in query.lower():
        speaker.Speak(f"Dont hesitate to call me again if you need any help... Have a good day..{name}")
        exit()

    elif "chat".lower() in query.lower():
        speaker.Speak("Chat mode on.")
        while True:
            query = takeCommand()
            if "exit" in query.lower():
                break
            chat(query)

    else:
        speaker.Speak("Sorry, I didn't get that. Please say again")
