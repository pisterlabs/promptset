import speech_recognition as sr
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

def window():
    app = QApplication([])
    w=QWidget()
    font = QFont('Times New Roman', 13)
    # label = QLabel(w)    #label.setPixmap(QPixmap("W:\\Projects\\Projects\\pyqt5\\11.jpg"))
    # label.setPixmap(QPixmap("W:\\Projects\\Projects\\pyqt5\\11.jpg"))
    # # label.setPixmap(QPixmap("W:\\Projects\\Projects\\pyqt5\\pexels-thisisengineering-3913025.jpg"))
    # label.setGeometry(0,0,370,600)
    # label.setStyleSheet(
    #     'width:100%;height:100%;'
    #)
    # w.setLayout()
    # label.setFont(font)
    # label.move(50, 50)
    
    auth=QLabel(w)
    name=QLabel(auth)
    na_text=QLineEdit(auth)
    passwl=QLabel(auth)
    passwt=QLineEdit(auth)
    
    
    display = QLabel(w)
    display.setGeometry(0, 0,400, 650)
    display.setStyleSheet('width:100%;height:100%;')
    display.setScaledContents(True)

    text=QMovie('QjoV.gif')
    # display.setStyleSheet('background:transparent;')
    display.setMovie(text)
    text.start()
    
    text_edit = QLineEdit(w)
    text_edit.setGeometry(10, 530, 290, 40)
    text_edit.setFont(QFont('Times New Roman',14))
    text_edit.setPlaceholderText('Search Here')
    text_edit.setStyleSheet('padding:10px;color:white;'
                         )

    button = QPushButton()
    button.setFont(font)
    button.setGeometry(250,530,50, 40)
    button.setIcon(QIcon('icons8-add-record-64.png'))
    # button.setStyleSheet('border-radius: 10px;background-color:blue;')
    button.setStyleSheet('cursor:pointer;border: none;background:white;border-radius: 20px;'
                         )
    Send = QPushButton(w)
    Send.setFont(font)
    # Send.setText('SEND')
    Send.setStyleSheet(
        'border-radius:20px;cursor:pointer;background-color:green;'
    )
    Send.setIcon(QIcon('icons8-send-30.png'))
    Send.setGeometry(310,530,50, 40)    

    mic = QPushButton(w)
    mic.setGeometry(310,480,50, 40)
    mic.setStyleSheet(
        'border-radius:20px;cursor:pointer;background-color:blue;'
    )
    mic.setIcon(QIcon('icons8-add-record-64.png'))

    def listen():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source)
        try:
            response = r.recognize_google(audio)
            print(f"You said: {response}")
            #text_edit.setText(response)
            Send.clicked.connect(command)

        except sr.UnknownValueError:
            print("Sorry, I did not understand.")
            # text_display.setText("Sorry, I did not understand.")

        except sr.RequestError:
            print("Sorry, I did not understand.")
            # text_display.setText("Sorry,  I did not understand.")

    def offlinelisten():
        from vosk import KaldiRecognizer,Model
        model=Model("W:\\Projects\\PyProjects\\assistants\\vosk-model-small-en-us-0.15")
        recognizer=KaldiRecognizer(model,16000)
        mic=pyaudio.PyAudio()
        stream=mic.open(format=pyaudio.paInt16,channels=1,rate=16000,input=True,frames_per_buffer=8192)
        try:
            data=stream.read(4096,exception_on_overflow=False)
            # if len(data)==0:
            #     break
            if recognizer.AcceptWaveform(data):
                rec=recognizer.Result()
                command2=json.loads(rec)
                print(command2['text'])
                #Send.toggled.connect(command)
        except:
            speak('Please I did not understand')    

    #weapp
    import sys
    import tkinter
    from pynput.keyboard import Key,Listener
    import logging
    import ctypes
    # from tkinter import *
    import turtle
    import pyautogui
    from PIL import ImageTk, Image
    import datetime
    import playsound
    import threading
    from datetime import datetime
    import json
    import time
    import webbrowser
    from playsound import playsound
    import pprint
    import getpass
    # import vosk
    import pyaudio
    # import playsound
    import subprocess as sp
    import wikipedia
    import speech_recognition as sr
    from pytube import YouTube
    import requests
    import cv2
    from cvzone.HandTrackingModule import HandDetector
    from cvzone.FaceDetectionModule import FaceDetector
    import math
    import pyttsx3
    import random
    from sketchpy import library as lib
    import calendar
    import os
    from pandas.io import clipboard
    from geopy.geocoders import Nominatim
    from geopy.distance import great_circle
    import geocoder
    from PIL import ImageGrab
    import numpy as np
    #normal conversation
    #solve mathematics
    #news
    #weather
    #temperature
    #face recognition
    #web scrapping
    #yt downloads
    #play alarm
    #make call
    #send message sms
    #email
    #jokes
    #advice
    #speech recognition
    #chempy
    #GUI
    #face recognition
    #mobile app
    #

    opening_text=['on it,sir','okay sir, i am working on it','a moment sir','getting it done,sir']

    commands=[
        'minimize',
        'maximize',
        'open settings',
        'open notepad',
        'close notepad',                    
        'open cmd',
        'open calculator',
        'calendar',
        'close notepad',
        'close cmd',
        'malware',
        'open camera',
        'close camera',
        'quote',
        'ip address',
        'date',
        'close calculator',
        'screenshot',
        'switch window',
        'shutdown',
        'refresh screen',
        'restart',
        'lock screen',
        'open youtube',
        'open stackoverflow',
        'open wikipedia',
        'draw',
        'search',
        'make note',
        'show note',
        'speak note',
        'play music',
        'cools',
        'play worship',
        'movie',
        'game',
        'close game',
        'music studio',
        'hack my phone',
        'volume up',
        'volume down',
        'mute',
        'unmute',
        'task manager',
        'download',
        'system info',
        'open obs studio',
        'close obs studio',
        'calculate',
        'quote',
        'hand detector',
        'face detector',
        'all detector'
    ]

    my_dictionary={
        'hello':'hi sir, how are you?',
        'hi':'hello sir, how are you?',
        'how are you':'I am as fit as fiddle and you',
        'how are you doing':'I am as fit as fiddle, and you?',
        'let talk business':'okay sir,ready for business',
        'how dare you':'sorry sir, I didnt mean offensive',
        'i am good':'Thank God',
        'can you':'Yes I can',
        'i love you':'I love you too',
        'good': 'I am also doing great sir',
        'great':'Thank God',
        'fine and you': 'also doing great sir',
        'fine': 'I am also doing great sir',
        'hey': 'hello, At your service,sir',
        'good and you': 'I am doing great',
        'whats up':'nothing much sir',
        'well done':'Thank you',
        'thanks': 'No thanks sir, after all you created me',
        'who are you': 'my name is Sunday',
        'what is your name': 'my name is Sunday',
        'who created you': 'I was created by PyDanny whose real name is Daniel Quansah',
        'i am fine and you':'I am also doing great sir!!',
        'how old are you':'I am nearly a year old',
        'who is pydanny':'PyDanny is a young, talented, self taught python developer, web developer and many more, who has aimed to achieve something great in future, thhus to be the greatest inventor ever.',
        'what can you do':'I can open and close programs and softwares, search for anything from google, youtube, wikipedia, stackoverflow, etc. Play music, movies, make note, create keylogger, give you random advice and jokes, and many more.',
        'how can you help me': 'I can open and close programs and softwares, search for anything from google, youtube, wikipedia, stackoverflow, etc.Play music, movies,make note,create keylogger,give you random advice and jokes.',
        'whats on':'Nothing much, sir!',
        'how was your day':'It was cool, sir and yours',
        'how was your night':'It was cool, sir and yours',
        'i am sad':'Oooh sir, please what happened',
        'tell me more about yourself':'I am Pi, a personal assistant program programmed by my creator PyDanny in pure Python',
        'tell me about yourself':'I am Pi, a personal assistant program programmed by my creator PyDanny in pure Python',
        'will you marry me':'No sir, I am only programmed to help you with some tasks',
        'can you marry me': 'No sir, I am only programmed to help you with some tasks',
        'sleep':'ok sir, you can wake me up anytime you want',
        'wake up':'hello sir, i am always at your service'
    }

    advice = ['Have the courage to live a life true to yourself, not the life others expect of you.',
            'Never attribute to malice that which can be adequately explained by stupidity.',
            '“There is nothing noble in being superior to your fellow man; true nobility is being superior to your former self.” Ernest Hemingway',
            'Don’t make decisions when you’re angry. Don’t make promises when you’re happy.',
            '“Never argue with a stupid person, they’ll drag you down to their level and beat you with experience.” Mark twain',
            'Only pack what you can carry yourself.',
            'Remember you’ll always regret what you didn’t do rather than what you did.',
            '“You’d worry less about what people think about you if you knew how seldom they do.” David Foster Wallace',
            'If you blame it on someone else, don’t expect it to get better.',
            '“You can be the ripest, juiciest peach in the world, but there will always be someone who hates peaches.” Dita von Teese',
            'If the grass is greener on the other side, there’s probably more manure there.',
            'Don’t give up what you want most for what you want now.',
            'With regards to the opposite sex: If you look hungry, you’ll starve.',
            '“Never let your sense of morals prevent you from doing what is right.” Isaac Asimov',
            'Strive to be the man you want your daughter to marry.',
            '“Remember only enemies speak the truth. Friends and lovers lie endlessly, caught in the web of duty.” Stephen King',
            'Never forget your car keys will change your car from one tonne of inert metal into one of the most deadly killing machines that has been invented.',
            'Wait 24 hours before getting mad and reacting about anything. If it doesn’t bother you in 24 hours time, it probably isn’t important enough to get mad over.',
            'Never make someone a priority who only makes you an option.',
            'Try not to take anything personally. No one thinks about you as much as you do.',
            '“If you want to know what a man’s like, take a good look at how he treats his inferiors, not his equals.” Sirius Black',
            'Figure out what you love to do, and then figure out how to get someone to pay you to do it.',
            'If you treat a woman like a queen, and she treats you like a jester, your princess is in another castle.',
            'Whenever something happens that makes you sad, ask yourself whether you’d still care about it when you’re ninety.',
            'Be persistent. When knowledge and ability aren’t enough, be persistent.',
            '“Life is scary. Get used to it. There are no magical fixes. It’s all up to you. So get up off your keister, get out of here, and go start doin’ the work. Nothing in this world that’s worth having comes easy.” Bob Kelso'
            'Smart girls like to hear they’re pretty, pretty girls like to hear that they’re smart.',
            'Happiness is a choice and everything else is a matter of perspective.']

    quote=['Youth has no age.',
        'The young do not know enough to be prudent, and therefore they attempt the impossible, and achieve it, generation after generation.',
        'Youth is not a question of years: one is young or old from birth.',
        'Use your youth so that you may have comfort to remember it when it has forsaken you, and not sigh and grieve at the account thereof.',
        'Youth is the season made for joys, love is then our duty.'
        'Never tell a young person that anything cannot be done.',
        'Your life, time, and brain should belong to you, not to an institution.',
        'How wonderful it is that nobody need wait a single moment before starting to improve the world.',
        'Aspire to inspire before we expire.',
        'Goodness is the only investment that never fails.',
        'Whatever you do, do it well.',
        'Winners never quit and quitters never win.',
        'You can\’t depend on your eyes when your imagination is out of focus.',
        'You have to grow from the inside out.',
        'Logic will get you from A to B. Imagination will take you everywhere.',
        'After a while, you learn to ignore the names people call you and just trust who you are.',
        'The world belongs to the energetic.',
        'We are all pretty bizarre. Some of us are just better at hiding it.',
        'Do, or do not. There is no try.',
        'I just wanna let them know that they didnt break me.',
        'Oh yes, the past can hurt. But you can either run from it, or learn from it.',
        'I feel infinite.',
        'High school is like the training wheels for the bicycle of real life.',
        'Youth is a gift of nature but age is a work of art.',
        'I define being the best as competing against the best there is out there and beating them.',
        'We are who we choose to be.',
        'Make each day your masterpiece.',
        'There was no respect for youth when I was young, and now that I am old, there is no respect for age, I missed it coming and going.',
        'This is what is missing in the youth today. This being able to dream and to change the world.',
        'In relative youth, we assume we will remember everything. Someone should urge the young to think otherwise.',
        'You are here to continually push forward and move forward. That is where I am at - I just want to always surprise people.',
        'But once we live all careless free; No cross to mar our love-lit bower.',
        'Life is like a box of chocolates, you never know what you\’re gonna get.',
        'I dont regret the things I\’ve done, but those I did not do.',
        'Me, I still believe in paradise. But now at least I know it\’s not some place you can look for because its not where you go.',
        'Life moves pretty fast. If you dont stop and look around once in a while, you could miss it.',
        'Enjoy the power of your beauty and your youth.',
        'The most lively young people become the best old people, not those who pretend to be as wise as grandfathers while they are still in school.',
        'Age is foolish and forgetful when it underestimates youth.',
        'Sometimes you face difficulties you are doing something wrong, but because you are doing something right.',
        'There is a fountain of youth: it is your mind, your talents, the creativity you bring to your life and the lives of the people you love.',
        'A smart man makes a mistake, learns from it, and never makes that mistake again.',
        'Its not what happens to you, but how you react to it that matters.',
        'It takes a very long time to become young.',
        'Only to children children sing, Only to youth will spring be spring.',
        'Youth is wasted on the young.',
        'The only person you should try to be better than, is the person you were yesterday.',
        'My grandfather once told me that there were two kinds of people: those who do the work and those who take the credit. He told me to try to be in the first group; there was much less competition.',
        'Joy of youth, dream of youth, blood of youth, mood of youth, clothe the world with colors golden, singing songs that never olden.',
        'Youth is that period when a young boy knows everything but how to make a living.',
        'Youth is the best time to be rich, and the best time to be poor.',
        'Youth is a blunder; manhood a struggle, 0ld age a regret.',
        'You are as old as your doubt, your fear, your despair.',
        'It always seems impossible until its done.',
        'Youth is something very new: twenty years ago, no one mentioned it.',
        'We would accomplish many more things if we did not think of them as impossible.',
        'One must learn by doing the thing; for though you think you know it, you have no certainty, until you try.',
        'You are only young once, and if you work it right, once is enough.',
        'The world is the great gymnasium where we come to make ourselves strong.',
        'When you reach the end of your rope, tie a knot in it and hang on.',
        'Cherish your visions and your dreams, as they are the children of your soul, the blueprints of your ultimate achievements.',
        'When you replace why is this happening to me with what is this trying to teach me? Everything shifts.',
        'The world’s biggest power is the youth and beauty of a woman.',
        'Everything started as nothing.',
        'It is better to be a young June-bug than an old bird of paradise.',
        'You always pass failure on the way to success.',
        'Young people need models, not critics.',
        'It is never too late to be what you might have been.',
        'Dont downgrade your dream just to fit your reality. Upgrade your conviction to match your destiny.',
        'Youth is the season of receptivity, and should be devoted to acquirement; and manhood of power--that demands an earnest application. Old age is for revision.',
        'Yesterday you said tomorrow. Just do it.',
        'Build yourself up to stand out and be recognized in a crowd.',
        'The secret of growing younger is counting blessings, not birthdays.',
        'It is the juvenile period of life when friendships are formed, and habits established, that will stick by one.',
        'A leader is a dealer in hope.',
        'In youth we learn; in age we understand.',
        'You don\’t have to hold a position in order to be a leader.',
        'Old minds have the power to create history, young minds have the power to change the history.',
        'To have long term success as a coach or in any position of leadership, you have to be obsessed in some way.',
        'Leaders are never satisfied; they continually strive to be better.',
        'The way to keep young is to keep your faith young. Keep your self-confidence young. Keep your hope young.',
        'The passions of the young are vices in the old.',
        'The young, free to act on their initiative, can lead their elders in the direction of the unknown.',
        'Unless someone like you cares a whole awful lot, nothing is going to get better.  Its not.',
        'Mans own youth is the worlds youth.',
        'We all know we\'re young. We\'re just trying to stay positive. We\'re making young mistakes.',
        'The young are not afraid of telling the truth.',
        'There is always some specific moment when we become aware that our youth is gone; but, years after, we know it was much later.',
        'Youth comes but once in a lifetime.',
        'It is time for parents to teach young people early on that in diversity there is beauty and there is strength.',
        'There is no doubt that creativity is the most important human resource of all.',
        'When you are young, everything feels like the end of the world, but its not; its just the beginning.',
        'Anyone who stops learning is old, whether at twenty or eighty. Anyone who keeps learning stays young.',
        'Youthfulness is about how you live, not when you were born.',
        'Life is not about waiting for the storms to pass... its about learning to dance in the rain.'
        ]

    p=pyttsx3.init('sapi5')
    voices=p.getProperty('voices')
    p.setProperty('voice',voices[1].id)
    p.setProperty('rate',170)
    p.setProperty('volume',1.0)

    def speak(audio):
        print('')
        # text_display.setText(audio)
        # text_display.setTextBackgroundColor(qGreen)
        p.say(audio)
        print(audio)
        p.runAndWait()

    #speak('hi sir')
    def greetings():
        hr=int(datetime.now().hour)
        if hr>=0 and hr<12:
            speak('Good morning sir!')
        elif hr>=12 and hr<16:
            speak('Good Afternoon sir!')
        else:
            speak('Good evening sir!')
    #greetings()

    def time_now():
        a=datetime.now().strftime('%H:%M')
        hr = int(datetime.now().hour)
        if hr>=6 and hr<12:
            speak(f'sir,  the time is {a}AM')
        elif hr>=12 and hr<=23:
            speak(f'sir,  the time is {a}PM')
        if hr>=0 and hr<6:
            speak(f'sir,  the time is {a}dawn')

    def date():
        d=datetime.now().date()
        ar = datetime.now().isoweekday()
        dic = {'1': 'Monday', '2': 'Tuesday', '3': 'Wednesday', '4': 'Thursday', '5': 'Friday', '6': 'Saturday'}
        aa = str(ar)
        if aa in dic:
            day=dic[aa]
            speak(f'sir, today is {day}, {d}')

    def take_command():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print('Listening....')
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print('Recognizing...')
            query = r.recognize_google(audio, language='en-in')
            if not 'exit' in query or 'stop' in query:
                speak(random.choice(opening_text))
            else:
                hour = datetime.datetime.now().hour
                if hour >= 21 and hour < 6:
                    speak("Good night sir, take care!")
                else:
                    speak('Have a good day sir!')
                exit()
        except Exception:
            speak('Unable to recognize your voice, sir')
            query = 'None'
        return query

    paths = {
        'notepad': "C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\Accessories\\Notepad.lnk",
        'calculator': "C:\\Windows\\System32\\calc.exe"
    }
    def open_notepad():
        os.startfile(paths['notepad'])

    def open_calculator():
        os.startfile(paths['calculator'])

    def find_my_ip():
        ip_address = requests.get('https://api64.ipify.org?format=json').json()
        return ip_address["ip"]

    def open_camera():
        sp.run('start microsoft.windows.camera:', shell=True)

    def open_cmd():
        os.system('start cmd')

    def search_on_wikipedia(query):
        results = wikipedia.summary(query, sentences=4)
        print(results)
        speak(results)
        return results

    def facedetector():
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        aa = FaceDetector()
        while (True):
            success, img = cap.read()
            imag = aa.findFaces(img)
            cv2.imshow('image', img)
            c = cv2.waitKey(1)
            if c == ord('q'):
                break

    def handdetector():
        cap = cv2.VideoCapture(0)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        while True:
            # Get image frame
            success, img = cap.read()
            # Find the hand and its landmarks
            hands, img = detector.findHands(img)  # with draw
            # hands = detector.findHands(img, draw=False)  # without draw

            if hands:
                # Hand 1
                hand1 = hands[0]
                lmList1 = hand1["lmList"]  # List of 21 Landmark points
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                centerPoint1 = hand1['center']  # center of the hand cx,cy
                handType1 = hand1["type"]  # Handtype Left or Right

                fingers1 = detector.fingersUp(hand1)

                if len(hands) == 2:
                    # Hand 2
                    hand2 = hands[1]
                    lmList2 = hand2["lmList"]  # List of 21 Landmark points
                    bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                    centerPoint2 = hand2['center']  # center of the hand cx,cy
                    handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                    fingers2 = detector.fingersUp(hand2)

                    # Find Distance between two Landmarks. Could be same hand or different hands
                    length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
                    #length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
            # Display
            cv2.imshow("Image", img)
            if cv2.waitKey(1)==ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def alldetector():
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280)
        cap.set(4, 720)
        detector = HandDetector(detectionCon=0.8, maxHands=2)
        face = FaceDetector(minDetectionCon=2)

        def main():
            while True:
                # Get image frame
                success, img = cap.read()
                # Find the hand and its landmarks
                hands, img = detector.findHands(img)
                faces = face.findFaces(img)
                # with draw
                # hands = detector.findHands(img, draw=False)  # without draw

                if hands:
                    # Hand 1
                    hand1 = hands[0]
                    lmList1 = hand1["lmList"]  # List of 21 Landmark points
                    bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
                    centerPoint1 = hand1['center']  # center of the hand cx,cy
                    handType1 = hand1["type"]  # Handtype Left or Right

                    fingers1 = detector.fingersUp(hand1)

                    if len(hands) == 2:
                        # Hand 2
                        hand2 = hands[1]
                        lmList2 = hand2["lmList"]  # List of 21 Landmark points
                        bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                        centerPoint2 = hand2['center']  # center of the hand cx,cy
                        handType2 = hand2["type"]  # Hand Type "Left" or "Right"

                        fingers2 = detector.fingersUp(hand2)

                        # Find Distance between two Landmarks. Could be same hand or different hands
                        length, info, img = detector.findDistance(lmList1[8][0:2], lmList2[8][0:2], img)  # with draw
                        # length, info = detector.findDistance(lmList1[8], lmList2[8])  # with draw
                # Display
                cv2.imshow("Image", img)
                c = cv2.waitKey(1)
                if c == ord('q'):
                    break

    def cartooning(img):
        import pyautogui as pg
        image=cv2.imread(img)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray=cv2.medianBlur(gray, 5)
        edges=cv2.adaptiveThreshold(gray,255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY,9,9)

        #CARTONATION
        color=cv2.bilateralFilter(image,9,250,250)
        cartoon=cv2.bitwise_and(color,color,mask=edges)
        cv2.imshow('image',image)
        cv2.imshow('edges',edges)
        cv2.imshow('cartoon',cartoon)
        c = cv2.waitKey(0)
        cv2.destroyAllWindows()

        while(True):
            a=pg.confirm('Are you sure about cartooning your picture', buttons=['Yes', 'No'])
            if a=='Yes':
                cartooning(img)
                break
            else:
                break

    def format_response(weather):
        for i in range(10):
            dt_txt = weather['list'][i]['dt']
            date = (datetime.fromtimestamp(dt_txt)).strftime('%d-%b-%Y %H:%M:%S')

            desc = weather['list'][i]['weather'][0]['description']

            temp_min = weather['list'][i]['main']['temp_min']

            temp_max = weather['list'][i]['main']['temp_max']

            final_str = 'Date: %s \nConditions: %s \nMin.Temperature(Celsius): %s\nMax.Temperature(Celsius): %s' %(date, desc, temp_min, temp_max)

            return(final_str)

    def get_weather(city):
        api_key = 'a4aa5e3d83ffefaba8c00284de6ef7c3'
        res = requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric").json()
        weather = res["weather"][0]["main"]
        temperature = res["main"]["temp"]
        feels_like = res["main"]["feels_like"]
        return weather, f"{temperature}℃", f"{feels_like}℃"

    def get_latest_news():
        NEWS_API_KEY = "17872d9bca7343e4866348761f3d3476"
        news_headlines = []
        res = requests.get(
            f"https://newsapi.org/v2/top-headlines?country=in&apiKey={NEWS_API_KEY}&category=general").json()
        articles = res["articles"]
        for article in articles:
            news_headlines.append(article["title"])
            speak(article, sep='\n')
        return news_headlines[:5]

    def note():
        speak('sir,please what should i write')
        nn = input(':')
        speak('sir,please should i include date and time?')
        datetimer = input(':')
        with open('log.txt', 'a') as file:
            time = datetime.now().strftime('%H:%M')
            date = datetime.now().date()
            if 'yes' in datetimer:
                file.write(f'on {date} at {time},you made a note saying {nn}')
                pyautogui.press('enter')
            elif 'no' in datetimer:
                file.write(f'you made a note saying {nn}')
                pyautogui.press('enter')
            speak('Note made successfully!')

    def open_note():
        os.startfile('D:\\Projects\\PyProjects\\assistants\\SUN\\log.txt')

    def play_englishsong():
        music_dir = 'D:\\English songs\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_englishsong():
        music_dir = 'D:\\English songs\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_hilife():
        music_dir = 'D:\\HI-LIFE\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_hilife():
        music_dir = 'D:\\HI-LIFE\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_gospel():
        music_dir = 'D:\\GOSPELS\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_gospel():
        music_dir = 'D:\\GOSPELS\\'
        songs = os.listdir(music_dir)
        #print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_worship():
        music_dir = 'D:\\Worship\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_worship():
        music_dir = 'D:\\Worship\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def Cools():
        music_dir = 'D:\\2015-05-05 COOLS.mp3'
        speak(f'Playing coools')
        title = os.startfile(music_dir )
    
    def playmo_song():
        music_dir = 'D:\\Lukas Graham - 7 Years [Official Live at House of Blues Dallas].mp3'
        speak(f'Playing Lukas Graham - 7 Years [Official Live at House of Blues Dallas].mp3')
        title = os.startfile(music_dir )

    def play_music():
        music_dir = 'D:\\P.L.A.Y.L.I.S.T\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_music():
        music_dir = 'D:\\P.L.A.Y.L.I.S.T\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_music1():
        music_dir = 'D:\\Afro-Dancehall\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_music1():
        music_dir = 'D:\\Afro-Dancehall\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_reggae():
        music_dir = 'D:\\Reggae\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def next_reggae():
        music_dir = 'D:\\Reggae\\'
        songs = os.listdir(music_dir)
        # print(len(songs))
        song = random.choice(songs)
        speak(f'Playing {song}')
        title = os.startfile(music_dir + song)

    def play_movie():
        movies_dir = 'D:\\Movies\\'
        movies = os.listdir(movies_dir)
        # print(len(movies))
        mov = movies[random.randint(0, len(movies)-1 )]
        speak(f'Playing {mov}')
        os.startfile(movies_dir + mov)

    def next_movie():
        movies_dir = 'D:\\Movies\\'
        movies = os.listdir(movies_dir)
        # print(len(movies))
        mov = movies[random.randint(0, len(movies)-1 )]
        speak(f'Playing {mov}')
        os.startfile(movies_dir + mov)

    def show_movie():
        os.startfile('D:\\Movies')

    def music_malware():
        dir = 'D:\\P.L.A.Y.L.I.S.T\\'
        songs=os.listdir(dir)
        title=songs[random.randint(0, len(songs) - 1)]
        print(title)
        all=dir+title
        os.startfile(all)
        time.sleep(220)

    def speak_note():
        with open('D:\\Projects\\Projects\\py\\log.txt', 'r') as ff:
            a = ff.read()
            speak(a)

    def game():
        path='C:\\Beach Buggy Racing.lnk'
        os.startfile(path)

    def football():
        path='C:\\Users\\Daniel\\Desktop\\Winning Eleven 9.lnk'
        os.startfile(path)

    def screenshot():
        time.sleep(1)
        pyautogui.hotkey('win','prtsc')

    def minimize_windows():
        time.sleep(1)
        pyautogui.hotkey('win','m')
        speak('all windows minimized')

    def keylogger():
        log_dir = ''
        logging.basicConfig(filename=(log_dir + 'keylogs.txt'), level=logging.DEBUG, format='%(asctime)s: %(message)s')

        def on_press(key):
            logging.info(str(key))

        def on_release(key):
            if key == Key.esc:
                return False

        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    def maximize_all():
        time.sleep(1)
        pyautogui.hotkey('win','shift','m')
        speak('all windows maximized')

    def draw():
        a=lib.rdj()
        a.draw()

    def barchart(index,labels,sizes):
        import matplotlib.pyplot as plot
        import pyautogui as pg
        import random
        #set up the data
        index1=[]
        speak('Please enter range')
        a=text_edit.text()
        for i in range(int(a)):
            index1.append(i)
        text_edit.clear()
        #print(index1)
        index=index1
        size=[]
        speak('Enter sizes label')
        for i in range(int(a)):
            r=int(text_edit.text())
            size.append(r)
            text_edit.clear()

        print(size)
        sizes=size

        label_s=[]
        for i in range(int(a)):
            speak('Please Enter Item label')
            label=str(text_edit.text)
            label_s.append(label)
            text_edit.clear()
        labels=label_s

        #setting up bar chart
        speak('Enter Vertical label')
        vert=str(text_edit.text())
        text_edit.clear()
        plot.ylabel(vert)
        speak('Enter Horizontal label')
        hori=str(text_edit.text())
        text_edit.clear()
        plot.xlabel(hori)
        plot.bar(index,sizes,tick_label=labels)
        plot.show()

        #pie chart set up
        plot.pie(sizes,labels=labels,autopct='%1.f%%',counterclock=False,startangle=105)
        plot.show()

    def calenda():
        speak('which year please?')
        year=int(input(':'))
        with open('new.txt','w') as calen:
            calen.write(calendar.calendar(year))
        speak(f'sir,please here is the calendar for {year}')
        os.startfile('new.txt')

    def quadratic_equation(a,b,c):
        rt = b ** 2 - (4 * a * c)
        ans1 =math.sqrt(rt)
        #print(ans1)
        top=-b+ans1
        down=2*a
        result1 = top / down
        rt2 = b ** 2 - (4 * a * c)
        ans2 =math.sqrt(rt2)
        #print(ans2)
        top2=-b-ans2
        down2=2*a
        result2 = top2 / down2
        answer='x1={} x2={}'.format(result1,result2)
        speak(f'Sir, the answer is {answer}')

    def calculate(a):
            time.sleep(5)
            pyautogui.click(x=352, y=149, clicks=1, interval=0, button='left')
            pyautogui.write(a, 0.5)
            pyautogui.press('enter')
            pyautogui.hotkey('ctrl', 'a')
            pyautogui.hotkey('ctrl','c')
            speak(f'{a} equals ' + clipboard.paste())

    def hackmyphone(ip_address):
        capture = cv2.VideoCapture(f"http:{ip_address}:8080/video")
        while True:
            success, frame = capture.read()
            cv2.imshow('Phone Camera', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        capture.release()
        cv2.destroyAllWindows()

    def alarm():
        #taking input from user
        speak('Sir, please take time to fill the alarm details')
        alarmDY=speak('Year')
        alarmDM=speak('Month')
        alarmDD=speak('Day')
        alarmH = speak('Hour')
        alarmM = speak('Minute>>>')
        amPm = speak('am/pm')
        mission=speak('Task')
        print(f"Waiting for the alarm @ {alarmH}:{alarmM}{amPm}")
        if (amPm == "pm"):
            alarmH = alarmH + 12
        #Current Date Time
        now = datetime.now()

        alarmDate=alarmDY,alarmDM,alarmDD
        #desired alarm time

        later = datetime(alarmDate,alarmH,alarmM,0)
        #calculating the difference between two time

        difference = (later - now)
        #difference in seconds

        total_sec=difference.total_seconds() 

        def alarm_func():
            for i in range(30):
                playsound('W:\\Projects\\Projects\\py\\beep-09.wav', True)
                speak(f'Sir, please it is time to {mission}')
            
        timer = threading.Timer(total_sec, alarm_func)
        timer.start()

    def loc(place):
        webbrowser.open("http://www.google.com/maps/place/" + place + "")
        geolocator = Nominatim(user_agent="myGeocoder")
        location = geolocator.geocode(place, addressdetails=True)
        target_latlng = location.latitude, location.longitude
        location = location.raw['address']
        target_loc = {'city': location.get('city', ''),
                    'state': location.get('state', ''),
                    'country': location.get('country', '')}
        current_loc = geocoder.ip('me')
        current_latlng = current_loc.latlng

        distance = str(great_circle(current_latlng, target_latlng))
        distance = str(distance.split(' ',1)[0])
        distance = round(float(distance), 2)

        return current_loc, target_loc, distance

    def my_location():
        ip_add = requests.get('https://api.ipify.org').text
        url = 'https://get.geojs.io/v1/ip/geo/' + ip_add + '.json'
        geo_requests = requests.get(url)
        geo_data = geo_requests.json()
        city = geo_data['city']
        state = geo_data['region']
        country = geo_data['country']
        #print(city,state,country)
        return city, state,country

    def record_screen():
        #Obtain image dimensions
        #Screen capture
        image = ImageGrab.grab()
        #Convert the object to numpy array
        img_np_arr = np.array(image)
        #Extract and print shape of array
        shape = img_np_arr.shape
        print(shape)

        #Create a video writer
        screen_cap_writer = cv2.VideoWriter('screen_recorded.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 50, (shape[1], shape[0]))
        #To View the screen recording in a separate window (OPTIONAL)
        #This is optional. Use the aspect ratio scaling if you wish to view the screen recording simultaneously
        #Low scale_by_percent implies smaller window
        scale_by_percent = 50 
        width = int(shape[1] * scale_by_percent / 100)
        height = int(shape[0] * scale_by_percent / 100)
        new_dim = (width, height)
        #Record the screen
        #Condition to keep recording as a video
        while True:
            #Capture screen
            image = ImageGrab.grab()
            #Convert to array
            img_np_arr = np.array(image)
            #OpenCV follows BGR and not RGB, hence we convert
            final_img = cv2.cvtColor(img_np_arr, cv2.COLOR_RGB2BGR)
            #Write to video 
            screen_cap_writer.write(final_img)
            #OPTIONAL: To view your screen recording in a separate window, resize and use imshow()
            '''
                If you choose to view the screen recording simultaneously,
                It will be displayed and also recorded in your video. 
            '''
            image = cv2.resize(final_img, (new_dim))
            cv2.imshow("image", image)
            #Stop and exit screen recoding if user presses 'e' (You can put any letter)
            if cv2.waitKey(1) == ord('q'):
                break

        #Release the created the objects
        screen_cap_writer.release()
        cv2.destroyAllWindows()

    def RealTimeCurrencyConversion():
        from forex_python.converter import CurrencyRates
        c = CurrencyRates()
        speak('Please how much')
        Amt=input('>>>')
        speak('From which currency  ["INR", "USD", "CAD", "CNY", "DKK", "EUR"]')
        from_currency=input('>>>')
        speak('To which currency  ["INR", "USD", "CAD", "CNY", "DKK", "EUR"]')
        to_currency=input('>>>')
        try:
            new_amt = c.convert(from_currency, to_currency, float(Amt))
            new_amount = float("{:.4f}".format(new_amt))
            speak(new_amount)
        except:
            speak('An error occurred, please check your internet connectivity')

    def command():
        query=text_edit.text()
        # query=input('\t\t\t\t\t\t\t\t\t\tBoss:').lower() or take_command().lower()
        #query=ques.get() or take_command.lower()
        if query in my_dictionary:
            text_edit.clear()
            speak(my_dictionary[query])

        elif 'minimize all' in query:
            text_edit.clear()
            minimize_windows()

        elif 'minimize' in query:
            text_edit.clear()
            pyautogui.hotkey('win','down')

        elif 'maximize all' in query:
            text_edit.clear()
            maximize_all()

        elif 'restore all' in query:
            text_edit.clear()
            maximize_all()

        elif 'maximize' in query:
            text_edit.clear()
            pyautogui.press('f11')

        elif 'restore' in query:
            text_edit.clear()
            pyautogui.press('f11')

        elif 'are you from' in query:
            speak('I am Proudly from Ghana')

        elif 'open settings' in query:
            text_edit.clear()
            pyautogui.hotkey('win','i')

        elif 'calendar' in query:
            speak('on it sir')
            text_edit.clear()
            calenda()

        elif 'alarm' in query:
            speak('on it sir')
            text_edit.clear()
            alarm()

        elif 'quadratic' in query:
            speak('solving quadratic equation in the form ax squared plus bx plus c equals 0')
            speak('Sir, please take time to fill in the values of a, b and c')
            a=int(text_edit.text)
            text_edit.clear()
            b=int(text_edit.text)
            text_edit.clear()
            c=int(text_edit.text)
            text_edit.clear()
            quadratic_equation(a,b,c)

        elif 'open notepad' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            open_notepad()

        elif 'close notepad' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im notepad.exe')

        elif 'open cmd' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            open_cmd()

        elif 'close cmd' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im cmd.exe')

        elif 'malware' in query:
            text_edit.clear()
            music_malware()

        elif 'keylogger' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            keylogger()
            speak('keylogger started!!!')

        elif 'let chat' in query:
            text_edit.clear()
            speak('ok sir, why not!!')

        elif 'cartoon image' in query:
            text_edit.clear()
            speak('sir, please input the imagelink')
            img = text_edit.text()
            cartooning(img)

        elif 'hand detector' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            handdetector()

        elif 'face detector' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            facedetector()

        elif 'detect face' in query:
            text_edit.clear()
            import cv2
            cascade_face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

            cap=cv2.VideoCapture(0)
            width,height=1280,720
            cap.set(3,width)
            cap.set(4,height)

            while(True):
                ret, img=cap.read()
                print(ret)
                g=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                F=cascade_face.detectMultiScale(g, 1.3, 6)
                for (x, y, w, h) in F:
                    cv2.rectangle(img,(x,y),(x + w,y +h),(0,255,0),4)
                cv2.imshow('img',img)
                k=cv2.waitKey(30) & 0xff
                if k==27:
                    break

            cap.release()
            cv2.destroyAllWindows()

        elif 'detect all' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            alldetector()

        elif 'open camera' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            open_camera()

        elif 'word' in query:
            text_edit.clear()
            speak('On it sir')
            pyautogui.hotkey('win', 'r')
            pyautogui.typewrite('winword')
            pyautogui.press('enter')
            speak('launching Microsoft Word 2019')
        
        elif 'excel' in query:
            text_edit.clear()
            speak('On it sir')
            pyautogui.hotkey('win', 'r')
            pyautogui.typewrite('excel')
            pyautogui.press('enter')
            speak('launching Microsoft Excel 2019')

        elif 'powerpoint' in query:
            text_edit.clear()
            speak('launching Microsoft Powerpoint 2019')
            os.startfile('C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\PowerPoint.lnk')

        elif 'close camera' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im camera.exe')

        elif 'date' in query:
            text_edit.clear()
            date()

        elif 'open calculator' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            open_calculator()

        elif 'calculate' in query:
            text_edit.clear()
            a = query.replace('calculate', '')
            open_calculator()
            calculate(a)

        elif 'close calculator' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im calculator.exe')

        elif 'screenshot' in  query:
            text_edit.clear()
            #speak(random.choice(opening_text))
            #screenshot()
            speak('tell me a name for the file')
            name=text_edit.text()
            time.sleep(3)
            img = pyautogui.screenshot() 
            img.save(f"{name}.png") 
            speak("screenshot saved")

        elif 'write' in query:
            query = query.replace("write", "")
            pyautogui.write(f"{query}",0.5)

        elif 'switch window' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            pyautogui.hotkey('alt','tab')
            speak('window switched')

        elif 'shutdown' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('shutdown /s')

        elif 'refresh screen' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            pyautogui.hotkey('win', 'm')
            pyautogui.rightClick(300, 300)
            pyautogui.press('Down')
            pyautogui.press('Down')
            pyautogui.press('Down')
            pyautogui.press('enter')
            pyautogui.hotkey('win', 'shift', 'm')

        elif 'restart' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('shutdown /r')

        elif 'lock screen' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('shutdown /l')

        elif 'quote' in query:
            text_edit.clear()
            speak(quote[random.randint(0,len(quote)-1)])

        elif 'advice' in query:
            text_edit.clear()
            speak(advice[random.randint(0,len(advice)-1)])

        elif 'open youtube' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'www.youtube.com'])

        elif 'download youtube video' in query:
            speak('on it, sir')
            text_edit.clear()
            speak('sir, please paste the video link')
            link=text_edit.text()
            yt=YouTube(link)
            try:
                speak('Downloading video')
                yd=yt.streams.get_highest_resolution()
                yd.download('W:\\','downloads by Py')            
                speak('video downloaded succesfully')

            except:
                speak('couldnt download video, please try again!')

        elif 'open google' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'www.google.com'])

        elif 'time' in query:
            text_edit.clear()
            time_now()

        elif 'read' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            text=clipboard.paste()
            speak(text)

        elif 'typeit' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            text=clipboard.paste()
            pyautogui.typewrite(text,0.5)
            speak('Done typing, Sir!!')

        elif 'weather' in query:
            text_edit.clear()
            ip_address = find_my_ip()
            city = requests.get(f"https://ipapi.co/{ip_address}/city/").text
            speak(f"Getting weather report for your city {city}")
            weather, temperature, feels_like = get_weather(city)
            speak(f"The current temperature is {temperature}, but it feels like {feels_like}")
            speak(f"Also, the weather report talks about {weather}")
            speak("For your convenience, I am printing it on the screen sir.")
            print(f"Description: {weather}\nTemperature: {temperature}\nFeels like: {feels_like}")

        elif "news" in query:
            text_edit.clear()
            try:
                news_res = get_latest_news()
                speak('Source: The Times Of Ghana')
                speak('Todays Headlines are..')
                for index, articles in enumerate(news_res):
                    pprint.pprint(articles['title'])
                    speak(articles['title'])
                    if index == len(news_res)-2:
                        break
                speak('These were the top headlines, Have a nice day Sir!!..')
            except:
                speak('Sir, please make sure you have internet connection')

        elif 'visit' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            web =  query.replace('visit', '')
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', f'{web}'])

        elif 'open stackoverflow' in  query:
            speak(random.choice(opening_text))
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'www.stackoverflow.com'])

        elif 'open wikipedia' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe', 'www.wikipedia.com'])

        elif 'search wikipedia' in  query:
            text_edit.clear()
            query =  query.replace('search wikipedia', '')
            speak(f'searching for {query} from wikipedia')
            search_on_wikipedia(query)

        elif 'draw' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            draw()

        elif 'search' in  query:
            text_edit.clear()
            a =  query.replace('search', '')
            speak(f'searching for {a} from google')
            sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
                    f'https://www.google.com/search?client=chrome-b-d&q={a}'])

        elif 'make note' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            note()

        elif 'show note' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            open_note()

        elif 'speak note' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            speak_note()

        elif 'play music' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_music()

        elif 'next music' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_music()

        elif 'play english' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_englishsong()

        elif 'next english' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_englishsong()

        elif 'play hilife' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_hilife()

        elif 'next hilife' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_hilife()

        elif 'harddisk' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.startfile('D:\\')

        elif 'play afromusic' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_music1()

        elif 'next afromusic' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_music1()

        elif 'play reggae' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_reggae()

        elif 'next reggae' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_reggae()

        elif 'stop music' in query:
            text_edit.clear()
            os.system('taskkill /f /im vlc.exe')

        elif 'stop movie' in query:
            text_edit.clear()
            os.system('taskkill /f /im vlc.exe')

        elif 'cools' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            Cools()
        elif 'mo-song' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            playmo_song()

        elif 'play gospel' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_gospel()

        elif 'next gospel' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_gospel()

        elif 'play worship' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_worship()

        elif 'next worship' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_worship()

        elif 'play movie' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            play_movie()

        elif 'next movie' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            next_movie()

        elif 'show movies' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            show_movie()

        elif 'plot' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            barchart()

        elif 'break' in query:
            text_edit.clear()
            speak('Please, how many minutes')
            time=input('>>')
            seconds=int(time)*60
            speak('Countdown started')
            pyautogui.countdown(seconds)
            print('Time Up',)
            #os.system('cls')
            for i in range(4):
                import playsound
                playsound.playsound('D:\\Projects\\Projects\\py\\beep-09.wav')
            speak('Time up, Sir. Time to get to work.')

        elif 'reminder' in query:
            text_edit.clear()
            speak('Please, how many minutes')
            time=text_edit.text()
            seconds=time*60
            speak('Countdown started')
            pyautogui.countdown(seconds)
            print('Time Up')
            #os.system('cls')
            for i in range(4):
                import playsound
                playsound.playsound('D:\\Projects\\Projects\\py\\beep-09.wav')
            speak('Time up, Sir. Time to get to work.')

        elif 'play game' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            game()

        elif 'football' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            football()

        elif 'close game' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im Beach Buggy Racing.lnk')

        elif 'music studio' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            sp.Popen(['C:\\Program Files\\Image-Line\\FL Studio 20\\FL64.exe'])

        elif 'hack my camera' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            speak('sir,please type your ip address')
            ipaddress = text_edit.text()
            speak('hacking your mobile phone camera')
            hackmyphone(ipaddress)

        elif 'volume up' in query:
            text_edit.clear()
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')
            pyautogui.press('volumeup')

        elif 'ip address' in query:
            text_edit.clear()
            try:
                find_my_ip()
            except:
                speak('an error occured, please make sure  you are connected to the internet.')

        elif 'volume down' in query:
            text_edit.clear()
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')
            pyautogui.press('volumedown')

        elif 'mute' in query:
            text_edit.clear()        
            pyautogui.press('volumemute')

        elif 'unmute' in query:
            text_edit.clear()
            pyautogui.press('volumemute')

        elif 'task manager' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            pyautogui.hotkey('ctrl','shift','esc')

        elif 'obs studio' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.startfile('C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\OBS Studio\\OBS Studio (64bit).lnk')

        elif 'hack my phone' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            speak('start streaming from ip webcam app on your phone and paste the ip address')
            ipaddress=input('Ip recommended:')
            hackmyphone(ipaddress)

        elif 'commands' in query:
            text_edit.clear()
            for word in commands:
                print(word+'\n')

        elif 'snake game' in query:
            text_edit.clear()
            speak(random.choice(opening_text))
            import snake_game
            snake_game()

        elif 'system info' in query:
            text_edit.clear()
            pyautogui.press('win')
            pyautogui.typewrite('system', 1)
            pyautogui.press('enter')
            #time.sleep(2)
            with open('D:\\Projects\\Projects\\system.txt', 'r') as file:
                a=file.read()
                speak(a)

        elif 'close obs studio' in  query:
            text_edit.clear()
            speak(random.choice(opening_text))
            os.system('taskkill /f /im OBS Studio (64bit).lnk')

        elif 'my picture' in query:
            text_edit.clear()
            speak('On it sir')
            os.startfile('mypic.JPG')
            speak('Here is PyDanny, the person wo created me')

        elif 'his picture' in query:
            text_edit.clear()
            speak('On it sir')
            os.startfile('mypic.JPG')
            speak('Here is PyDanny, the person wo created me')

        elif "where is" in query:
            text_edit.clear()
            place = query.split('where is ', 1)[1]
            current_loc, target_loc, distance =loc(place)
            city = target_loc.get('city', '')
            state = target_loc.get('state', '')
            country = target_loc.get('country', '')
            # time.sleep(1)
            try:
                if city:
                    res = f"{place} is in {state} state and country {country}. It is {distance} km away from your current location"
                    #print(res)
                    speak(res)
                else:
                    res = f"{state} is a state in {country}. It is {distance} km away from your current location"
                    #print(res)
                    speak(res)
            except:
                res = "Sorry sir, I couldn't get the co-ordinates of the location you requested. Please try again"
                #print(res)
                speak(res)

        elif "current location" in query or "where am i" in query:
            text_edit.clear()
            try:
                city, state, country = my_location()
                print(city, state, country)
                speak(f"You are currently in {city} city which is in {state} state and country {country}")
            except Exception as e:
                print(e)
                speak("Sorry sir, I coundn't fetch your current location. Please try again")

        elif 'screen recorder' in query:
            text_edit.clear()
            speak('Starting screen recorder')
            record_screen()

        elif 'record screen' in query:
            text_edit.clear()
            speak('Starting screen recorder')
            record_screen()

        elif 'instagram' in query:
            text_edit.clear()
            speak('opening your instagram, sir')
            webbrowser.open('www.instagram.com/pydanny21')

        elif 'telegram' in query:
            text_edit.clear()
            speak('opening your telegram, sir')
            webbrowser.open('https://t.me/pydanny21')

        elif 'github' in query:
            text_edit.clear()
            speak('opening your gitub, sir')
            webbrowser.open('www.github.com/pydanny21')

        elif 'facebook' in query:
            text_edit.clear()
            speak('opening your facebook, sir')
            webbrowser.open('www.facebook.com')

        elif 'website' in query:
            text_edit.clear()
            speak('opening your portfolio website, sir')
            webbrowser.open('www.sites.google.com/view/pyportfolio')

        elif 'search youtube' in query:
            text_edit.clear()
            video=query.replace('search youtube','')
            try:
                import pywhatkit
                kit.playonyt(video)
            except:
                speak('Sir, please make sure you have a stable internet connection')

        elif 'play on youtube' in query:
            text_edit.clear()
            video=query.replace('play on youtube','')
            try:
                import pywhatkit
                kit.playonyt(video)
            except:
                speak('Sir, please make sure you have a stable internet connection')

        elif 'send whatsapp message to' in query:
            text_edit.clear()
            speak('Sir,please what should I say')
            message=text_edit.text()
            number=int(query.replace('send whatsapp message to',''))
            try:
                import pywhatkit as kit
                kit.sendwhatmsg_instantly(number,message)
            except:
                speak('Sir, please make sure you have a stable internet connection')

        elif 'convert currency' in query:
            text_edit.clear()
            RealTimeCurrencyConversion()

        elif 'virtual dj' in query:
            text_edit.clear()
            speak('Opening virtual dj 2018')
            os.startfile('C:\\Program Files (x86)\\VirtualDJ\\virtualdj8.exe')

        elif 'designer' in query:
            text_edit.clear()
            speak('Starting QT Designer for you,Sir')
            os.startfile('C:\\Users\\Daniel\\AppData\\Roaming\\Python\\Python39\\site-packages\\qt5_applications\\Qt\\bin\\designer.exe')

        elif 'say' in query:
            text_edit.clear()
            say=query.replace('say','')
            speak(say)
        # else:
        #     speak('sorry sir . Please I did not find that in my database.')
        #     speak(f'searching for {query} from google')
        #     sp.Popen(['C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        #             f'https://www.google.com/search?client=chrome-b-d&q={query}'])
        # #speak('sir,please what should i do again.')

        elif 'code' in query:
            import openai
            conversation=text_edit.text()
            text_edit.clear()
            try:
                openai.api_key = "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"
                response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=conversation,
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                speak(response.choices[0].text)
            except:
                speak('Please I did not understand.')
                
        elif 'open chrome' in query:
            os.startfile('C:\Program Files\Google\Chrome\Application\chrome.exe')

        elif 'close chrome' in query:
            os.system("taskkill /f /im chrome.exe")

        elif 'maximize this window' in query:
            pyautogui.hotkey('alt', 'space')
            time.sleep(1)
            pyautogui.press('x')
    
        
        elif 'youtube search' in query:
            query = query.replace("youtube search", "")
            pyautogui.hotkey('alt', 'd')
            # time.sleep(1)
            pyautogui.press('tab')
            pyautogui.press('tab')
            pyautogui.press('tab')
            pyautogui.press('tab')
            time.sleep(1)
            pyautogui.write(f"{query}", 0.1)
            pyautogui.press('enter')
            
        elif 'open new window' in query:
            pyautogui.hotkey('ctrl', 'n')
            
        elif 'open incognito window' in query:
            pyautogui.hotkey('ctrl', 'shift', 'n')
            
        elif 'minimise this window' in query:
            pyautogui.hotkey('alt', 'space')
            # time.sleep(1)
            pyautogui.press('n')
            
        elif 'open history' in query:
            pyautogui.hotkey('ctrl', 'h')
        
        elif 'open downloads' in query:
            pyautogui.hotkey('ctrl', 'j')
        
        elif 'previous tab' in query:
            pyautogui.hotkey('ctrl', 'shift', 'tab')
        
        elif 'next tab' in query:
            pyautogui.hotkey('ctrl', 'tab')
        
        elif 'close tab' in query:
            pyautogui.hotkey('ctrl', 'w')
        
        elif 'close window' in query:
            pyautogui.hotkey('ctrl', 'shift', 'w')
        
        elif 'clear browsing history' in query:
            pyautogui.hotkey('ctrl', 'shift', 'delete')
        
        elif 'close chrome' in query:
            os.system("taskkill /f /im chrome.exe")

        elif "scroll down" in query:
            pyautogui.scroll(1000)

        else:
            speak('could you please come again?')
            text_edit.clear()

    passw='PyDanny@21'

    speak('Hello Sir, Welcome!!')
    speak('Sir, Please Authenticate')
    verify= getpass.getpass("\t\t\t\t\t\t\t\t\t\tPassword: ")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
    if verify==passw:
        # speak('Activating system configuration.')
        # speak('Pulling all necessary requests. ')
        # time.sleep(5)
        speak('I am online and ready')
        #speak('To use speech recognition, you must use an internet connection')
        greetings()
        # speak('I am Pi, your personal assistant')
        speak('please,how may i help you.')

    else:
        speak('Wrong Password')
        speak('I could not start the program')
        speak('Shutting down the system')
        # import playsound
        # # import time
        # for i in range(4):
        #     playsound.playsound('W:\\Projects\\Projects\\py\\beep-09.wav')
        sys.exit()

    # button.clicked.connect(offlinelisten)
    button.clicked.connect(listen)
    # Send.clicked.connect(isClicked)
    Send.clicked.connect(command)
    text_edit.editingFinished.connect(command)
    # text_display.setText(text_edit.text())
    w.setWindowTitle("PyAssistant")
    w.setStyleSheet(
        'background:black;display:flex;width:100%;height:100%;margin:0px;padding:0px;'
    )
    w.setGeometry(100, 100,370, 600)
    w.show()
    app.exec_()

if __name__ == '__main__':
    window()