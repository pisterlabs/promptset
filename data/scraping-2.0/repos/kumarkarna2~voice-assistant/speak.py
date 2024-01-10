"""library for text to speech"""
import webbrowser
import os
import smtplib
import datetime
import pyttsx3
import speech_recognition as sr
import wikipedia
import openai


# Initialize the API key
openai.api_key = "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"


def responses(prompt):
    """This function takes the input from user
    and returns the response from the API"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        # engine="text-curie-001",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text  # type: ignore
    return message


messages = [
    {
        "role": "system",
        "content": "This is a chatbot that only answer questions related to Karna Kumar Chaudhary. For questions not related to Vivien Chua, reply with Sorry, I do not know.",
    },
    {"role": "user", "content": "Who is Karna Kumar Chaudhary?"},
    {
        "role": "assistant",
        "content": "Karna Kumar Chaudhary is cse undergraduate student at Jaypee University of Information Technology, Waknaghat, Solan, Himachal Pradesh, India. He is a frontend developer and a competitive programmer. He is also a machine learning and AI enthusiast. He is currently working on a project named virtual assistant using python.",
    },
]


def generate_response(prompt):
    if prompt:
        messages.append({"role": "user", "content": prompt})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content  # type: ignore
        messages.append({"role": "assistant", "content": reply})

    return reply  # type: ignore


def sendEmail(to, content):
    """function to send email"""
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.ehlo()
    server.starttls()
    server.login("201172@juitsolan.in", "password")
    server.sendmail("201172@juitsolan.in", to, content)
    server.close()


def speak(audio):
    """simple function to speak the text"""
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.say(audio)
    engine.runAndWait()


def wish():
    """wish the user"""
    # speak("Welcome back sir!")
    hour = int(datetime.datetime.now().hour)
    if hour >= 4 and hour < 12:
        print("Good morning sir!")
        speak("Good morning sir!")
    elif hour >= 12 and hour < 18:
        print("Good afternoon sir!")
        speak("Good afternoon sir!")
    elif hour >= 18 and hour < 20:
        print("Good evening sir!")
        speak("Good evening sir!")
    else:
        print("How can i help you or should i say Good night sir!")
        speak("How can i help you or should i say Good night sir!")


def takeCommand():
    """it takes voice input from the user and returns output as text"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
        r.energy_threshold = 1000
        r.adjust_for_ambient_noise(source, duration=1)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language="en-in")
        print(f"Me : {query}\n")

    except Exception:  # pylint: disable=broad-except
        # print(e)
        print("Say that again please...")
        return "None"
    return query


if __name__ == "__main__":
    wish()
    while True:
        uquery = takeCommand().lower()  # type: ignore

        if "wikipedia" in uquery:
            print("Searching wikipedia...")
            speak("Searching wikipedia...")
            # replace wikipedia with empty string
            uquery = uquery.replace("wikipedia", "")
            results = wikipedia.summary(uquery, sentences=2)
            print("According to wikipedia")
            speak("According to wikipedia")
            print(results)
            speak(results)

        elif "open youtube" in uquery:
            print("Opening youtube...")
            speak("Opening youtube...")
            webbrowser.open("youtube.com")

        elif "open google" in uquery:
            print("Opening google...")
            speak("Opening google...")
            webbrowser.open("google.com")

        elif "play music" in uquery:
            m_dir = "D:\\music"
            songs = os.listdir(m_dir)
            print("Playing music...")
            speak("Playing music...")
            os.startfile(os.path.join(m_dir, songs[0]))

        elif "time" in uquery:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"Sir, the time is {strTime}")
            speak(f"Sir, the time is {strTime}")

        elif "open code" in uquery:
            loc = "C:\\Users\\karna\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
            print("Opening visual studio code...")
            speak("Opening visual studio code...")
            os.startfile(loc)

        elif "open gmail" in uquery:
            print("Opening gmail...")
            speak("Opening gmail...")
            webbrowser.open("gmail.com")

        elif "email to karna" in uquery:
            try:
                print("What should i say?")
                speak("What should i say?")
                content = takeCommand()
                to = "201172@juitsolan.in"
                sendEmail(to, content)
                print("Email has been sent!")
                speak("Email has been sent!")
            except Exception:  # pylint: disable=broad-except
                print("Unable to send email")
                speak("Unable to send email")

        elif "quit" in uquery:
            print("Do you need anything else sir? or should i go for a nap? ")
            speak("Do you need anything else sir? or should i go for a nap? ")
            qstr = takeCommand().lower()  # type: ignore
            if "no" in qstr or "nope" in qstr or "nah" in qstr or "no thanks" in qstr:
                print("Ok sir, i am going for a nap")
                speak("Ok sir, i am going for a nap")
                exit()
            elif "yes" in qstr or "yeah" in qstr or "sure" in qstr or "yup" in qstr:
                print("Ok sir, what can i do for you?")
                speak("Ok sir, what can i do for you?")
                continue
            else:
                print("Sorry sir, i didn't get you")
                speak("Sorry sir, i didn't get you")

        elif "sign out" in uquery or "log out" in uquery or "log off" in uquery:
            print("Do you wish to log out your computer ? (yes / no): ")
            speak("Do you wish to log out your computer ? (yes / no): ")
            logout = takeCommand().lower()  # type: ignore
            if logout == "no":
                exit()
            else:
                os.system("shutdown /l")

        elif "shutdown" in uquery:
            print("Do you wish to shut down your computer ? (yes / no): ")
            speak("Do you wish to shut down your computer ? (yes / no): ")
            shutdown = takeCommand().lower()  # type: ignore
            if shutdown == "no":
                exit()
            else:
                os.system("shutdown /s /t 1")

        elif "restart" in uquery:
            print("Do you wish to restart your computer ? (yes / no): ")
            speak("Do you wish to restart your computer ? (yes / no): ")
            res = takeCommand().lower()  # type: ignore
            if res == "no":
                exit()
            else:
                os.system("shutdown /r /t 1")

        elif "sleep" in uquery:
            print("Do you wish to put your computer to sleep ? (yes / no): ")
            speak("Do you wish to put your computer to sleep ? (yes / no): ")
            sleep = takeCommand().lower()  # type: ignore
            if sleep == "no":
                exit()
            else:
                os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")

        else:
            print("Searching...")
            speak("Searching...")
            prompt = uquery
            # message = responses(prompt)
            message = generate_response(prompt)
            print(message)
            speak(message)
