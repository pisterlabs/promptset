import random

import speech_recognition as sr
import openai
from openai import OpenAI
import wikipedia
import os
import webbrowser as wb
import win32com.client
import datetime
import smtplib
# import nepali_date_converter as npdate
import nepali
import sys

import pywhatkit as kit
from config import apikey

chatstr = ""


def okboss():
    speakfunc("Ok ,Boss")


def SendEmail(to, content):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.login('', '')
    server.sendmail('kushalbaral101@gmail.com', f'{to}', f'{content}')
    server.close()


def chatfunc(query):
    global chatstr
    key = apikey
    chatstr += f"Kushal: {query} \nASTRA: "
    print(chatstr)

    from openai import OpenAI
    client = OpenAI(api_key=key)

    response = client.completions.create(
        model="text-davinci-003",
        prompt=chatstr,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    chatstr += f"{response.choices[0].text}\n"
    return response.choices[0].text


def aifunc(prompt):
    text = f"{prompt} \n ------------------------------------\n\n"
    key = apikey
    from openai import OpenAI
    client = OpenAI(api_key=key)

    response = client.completions.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    text_only = response.choices[0].text

    print(f"ASTRA :{text_only}")
    text += response.choices[0].text
    if not os.path.exists("ANSbyASTRA"):
        os.mkdir("ANSbyASTRA")
    with open(f"ANSbyASTRA/{prompt[7:]}.txt", "w") as f:
        f.write(text)
    speakfunc("Do you want me to read this")
    ans = takeCommand()
    if "yes" or "of course" or "sure" or "why not" or "glad if you do so" in ans.lower():
        okboss()
        speakfunc(text_only)
    else:
        okboss()
    speakfunc("Do u wanna save the answer in File as a text format")
    query = takeCommand()
    if "yes" in query.lower():
        okboss()
        if not os.path.exists("ANSbyASTRA"):
            os.mkdir("ANSbyASTRA")
        with open(f"ANSbyASTRA/{prompt[7:]}.txt", "w") as f:
            f.write(text)
        speakfunc("I have  saved the answer for your convienence .You can check if needed later")
    else:
        okboss()


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        # r.pause_threshold = 0.8
        try:
            print("ASTRA: Recognizing")
            text = r.recognize_google(audio,language="en-US")
            print(f"Kushal :{text}")
            return text
        except Exception as e:
            print("ASTRA > Sorry Boss,I dont get it? ")
            return " "


def speakfunc(text):
    print(f"ASTRA: {text}")
    txt = text
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak(txt)


if __name__ == '__main__':

    speakfunc("HI,BOSS")
    x = True
    while x == True:
        print("ASTRA > Listening...")
        query = takeCommand()
        if "oshin" in query.lower():
            speakfunc("Hi,What can i do for you boss")
        if "open music".lower() in query.lower():
            speakfunc(f"Opening music Boss ...")
            musicpath = "C:\music\jg.mp3"
            os.startfile(musicpath)
            x = False
        sitelist = [["youtube", "https://youtube.com"], ["google", "https://google.com"],
                    ["wikipedia", "https://wikipedia.com"], ["facebook", "https://facebook.com"]]

        for site in sitelist:
            if f"open {site[0]}".lower() in query.lower():
                speakfunc(f"Opening {site[0]} Boss ...")
                wb.open(site[1])
                x = False

        if "the time".lower() in query.lower():
            strtime = datetime.datetime.now().strftime("%H:%M")
            speakfunc(f"ASTRA > Boss,Time is {strtime}")
            x = False

        applist = [["vs code", r"C:\Users\Public\Desktop\Vscode.lnk"],
                   ["git bash", r"C:\Users\Public\Desktop\git bash.lnk"],
                   ["file manager", r"C:\Users\Public\Desktop\file manager.lnk"],
                   ["chrome", r"C:\Users\ACER.KUSHAL101\Desktop\GoogleChrome.lnk"],
                   ["microsoft edge", r"C:\Users\Public\Desktop\Microsoft Edge.lnk"],
                   ["chat gpt", r"C:\Users\Public\Desktop\ChatGPT.lnk"],
                   ["settings", r"C:\Users\Public\Desktop\setting.lnk"], ]

        for app in applist:
            if f"open {app[0]}" in query.lower():
                speakfunc(f"OK BOSS,Opening {app[0]}")
                os.system(f"{app[1]}")
                x = False
        if "clear" in query.lower():
            okboss()
            chatstr = ""

        elif "thank you" in query.lower():
            speakfunc("OK,Boss Have a great day")
            x = False
        elif "play song on youtube" in query.lower():
            speakfunc("Tell me the name of song you wanna play")
            query = takeCommand()
            print(f"Kushal: {query}")
            kit.playonyt(query)
            okboss()
            # x=False
        elif "send email" in query.lower():
            try:
                # speakfunc("Tell me Whom to send mail")
                # to=takeCommand()
                to = "kusalbaral101@gmail.com"
                speakfunc("What to say boss")
                content = takeCommand()
                speakfunc(f"Boss confirm You wanna send mail to {to} saying {content}")
                ans = takeCommand()
                if "ok" in ans.lower():
                    okboss()
                    SendEmail(to, content)
                    speakfunc("Message sent")
                else:
                    okboss()
                    speakfunc("Email not send")
            except Exception as e:
                speakfunc("Some error have Occured {e}")
        elif "terminate tab" in query.lower():
            speakfunc("Tell me name of app you wanna close")
            tab = takeCommand()
            print(tab)
            os.system("taskkill /IM {}.exe /F")
            okboss()

        if "please".lower() in query.lower():
                       aifunc(prompt=query)
        else:
            speakfunc(chatfunc(query))
