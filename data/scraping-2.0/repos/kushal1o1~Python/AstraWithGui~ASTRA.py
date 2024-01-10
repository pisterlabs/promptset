import random
import time
import pyautogui as pg
import speech_recognition as sr
import openai
from PyQt5.QtCore import QCoreApplication
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
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import QTimer, QTime, QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from ASTRAUI.ASTRA1 import Ui_MainWindow

import pythoncom
import pywhatkit as kit
from config import apikey
x=0
hr = int(time.strftime('%H'))
if (hr > 12):
    hr = hr - 12
hr=hr
min = int(time.strftime('%M'))
print(f"Time is {hr}:{min}")
hr = int(time.strftime('%H'))
chatstr = "                            I am virtual assistant ASTRA \n                                       ------------------------------------\n                                       SAY WAKE UP TO CONTINUE\n                                       ------------------------------------\n"
chatstr2=""
action="ASTRA"
def returntext(text):
        chatstr=text

def returnaction(act):
        global action
        action=act       
class MainThread(QThread):
    def __init__(self):
        super(MainThread, self).__init__()

    def run(self):
        global returnaction

        pythoncom.CoInitialize()
        
        self.speakfunc("Say wakeup to continue !!")
        returnaction("Sleeping...")
        while True:
            global x
            global chatstr
            self.query=self.takeCommand()
            if 'wake up'in self.query.lower():
                  x=1
                  chatstr=""
                  self.MAIN()
            returnaction("Sleeping...")


    def okboss(self):
        self.speakfunc("Ok ,Boss")

    def wish(self):
        global hr 
        global min
        global chatstr
        hr = int(time.strftime('%H'))
        if (hr > 12):
            hr = hr - 12
        min = int(time.strftime('%M'))
        print(f"Time is {hr}:{min}")
        hr = int(time.strftime('%H'))

        if (hr >= 0 and hr < 12):

            self.speakfunc(f"Good Morning ,Boss It's {hr} {min} AM")
            returntext(f"Good Morning ,Boss It's {hr} {min} AM")
            # return self.speakfunc("I am Your virtual assistant ASTRA \nPlease tell me how can I help you ?")
        elif (hr >= 12 and hr < 17):
            if (hr > 12):
                hr = hr - 12
            self.speakfunc(f"Good Afternoon ,Boss It's {hr} {min} PM")
            # return self.speakfunc("I am Your virtual assistant ASTRA\n Please tell me how can I help you ?")
        else:
            if (hr > 12):
                hr = hr - 12
            self.speakfunc(f"Good Evening ,Boss It's {hr} {min} PM")
            # return self.speakfunc("I am Your virtual assistant ASTRA\n Please tell me how can I help you ?")

    def SendEmail(self, to, content):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.login('', '')
        server.sendmail('kushalbaral101@gmail.com', f'{to}', f'{content}')
        server.close()

    def chatfunc(self, query):

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

    def aifunc(self, prompt):
        global chatstr
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
        chatstr = response.choices[0].text

        # chatstr=f"ASTRA :{text_only}"
        print(f"ASTRA :{chatstr}")
        returntext(f"{chatstr}\n")
        text += response.choices[0].text
        if not os.path.exists("ANSbyASTRA"):
            os.mkdir("ANSbyASTRA")
        with open(f"ANSbyASTRA/{prompt[7:]}.txt", "w") as f:
            f.write(text)
        self.speakfunc("Do you want me to read this")
        ans = self.takeCommand()
        if "yes" or "of course" or "sure" or "why not" or "glad if you do so" in ans.lower():
            self.okboss()
            self.speakfunc(chatstr)
        else:
            self.okboss()
            self.asktoquit()
        self.speakfunc("Do u wanna save the answer in File as a text format")
        self.query = self.takeCommand()
        if "yes" or "of course" or "sure" or "why not" or "glad if you do so"  in self.query.lower():
            self.okboss()
            if not os.path.exists("ANSbyASTRA"):
                os.mkdir("ANSbyASTRA")
            with open(f"ANSbyASTRA/{prompt[7:]}.txt", "w") as f:
                f.write(text)
            self.speakfunc("I have  saved the answer for your convienence .You can check if needed later")
            self.asktoquit()
        else:
            self.okboss()
            self.asktoquit()

    def takeCommand(self):
        global action
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            # r.pause_threshold = 0.8
            try:
                print("ASTRA: Recognizing")
                returnaction("Recognizing")
                text = r.recognize_google(audio, language="en-US")
                print(f"Kushal :{text}")
                # returnaction("")
                returntext(f"Kushal :{text}")
                return text
            except Exception as e:
                print("ASTRA > Sorry Boss,I dont get it? ")
                return ""
                

    def speakfunc(self, text):
        global action
        print(f"ASTRA: {text}")
        
        txt = text
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        returntext(f"ASTRA: {text}")
        returnaction("Speaking")
        speaker.Speak(txt)

    def asktoquit(self):
        global x
        global chatstr
        global returnaction
        self.speakfunc("Is there anything else I can do for you?")
        self.query=self.takeCommand()
        if "no"in self.query.lower() or "not at all" in self.query.lower() or "never mind" in self.query.lower() or "bye bye" in self.query.lower() or "goodbye"in self.query.lower() or "see you soon"in self.query.lower()or "see you later"in self.query.lower() or "nope" in self.query.lower() in self.query.lower():
            self.speakfunc("Going to sleep boss! have a nice day")
            chatstr='                    Sleeping...\n                    Say WAKEUP To run'
            returnaction("Sleeping...")
            x=0
        else:
            x=1
        
            

             


    def MAIN(self):
        global chatstr
        chatstr=""
        while x==1:
            # self.wish()
            print("ASTRA > Listening...")
            returntext("ASTRA > Listening...")
            returnaction("Listening")
            self.query = self.takeCommand()
            if "astra" in (self.query).lower():
                self.speakfunc("hmm ,GoOn I'm listening Boss")
            if "open music".lower() in self.query.lower():
                self.speakfunc(f"Opening music Boss ...")
                musicpath = "C:\music\jg.mp3"
                os.startfile(musicpath)
                self.asktoquit()
            sitelist = [["youtube", "https://youtube.com"], ["google", "https://google.com"],
                        ["wikipedia", "https://wikipedia.com"], ["facebook", "https://facebook.com"]]

            for site in sitelist:
                if f"open {site[0]}".lower() in self.query.lower():
                    self.speakfunc(f"Opening {site[0]} Boss ...")
                    wb.open(site[1])
                    self.asktoquit()

            if "the time".lower() in self.query.lower():
                strtime = datetime.datetime.now().strftime("%H:%M")
                self.speakfunc(f"ASTRA > Boss,Time is {strtime}")
                self.asktoquit()

            applist = [["vs code", r"C:\Users\Public\Desktop\Vscode.lnk"],
                       ["git bash", r"C:\Users\Public\Desktop\git bash.lnk"],
                       ["file manager", r"C:\Users\Public\Desktop\file manager.lnk"],
                       ["chrome", r"C:\Users\ACER.KUSHAL101\Desktop\GoogleChrome.lnk"],
                       ["microsoft edge", r"C:\Users\Public\Desktop\Microsoft Edge.lnk"],
                       ["chat gpt", r"C:\Users\Public\Desktop\ChatGPT.lnk"],
                       ["settings", r"C:\Users\Public\Desktop\setting.lnk"], ]

            for app in applist:
                if f"open {app[0]}" in self.query.lower():
                    self.speakfunc(f"OK BOSS,Opening {app[0]}")
                    os.system(f"{app[1]}")
                    self.asktoquit()

            if "clear the screen" in self.query.lower():
                chatstr=""
                self.okboss()
                self.asktoquit()
                

            elif "play song on youtube" in self.query.lower():
                self.speakfunc("Tell me the name of song you wanna play")
                self.query = self.takeCommand()
                print(f"Kushal: {self.query}")
                kit.playonyt(self.query)
                self.okboss()
                self.asktoquit()
               
            elif "send email" in self.query.lower():
                try:
                    # speakfunc("Tell me Whom to send mail")
                    # to=takeCommand()
                    to = "kusalbaral101@gmail.com"
                    self.speakfunc("What to say boss")
                    content = self.takeCommand()
                    self.speakfunc(f"Boss confirm You wanna send mail to {to} saying {content}")
                    ans = self.takeCommand()
                    if "ok" in ans.lower():
                        self.okboss()
                        self.SendEmail(to, content)
                        self.speakfunc("Message sent")
                    else:
                        self.okboss()
                        self.speakfunc("Email not send")
                except Exception as e:
                    self.speakfunc("Some error have Occured {e}")
            elif "close tab" in self.query.lower():
                self.speakfunc("Tell me name of app you wanna close")
                tab = self.takeCommand()
                print(tab)
                os.system("taskkill /IM {}.exe /F")
                self.okboss()
                self.asktoquit()

            elif "please".lower() in self.query.lower():
                self.aifunc(prompt=self.query)
            # elif "no" or "no thanks" in self.query.lower():
            #     speakfunc("ok ,BYE boss")
            #     exit(0)
            elif "take a screenshot" in self.query.lower():
                self.okboss()
                img = pg.screenshot()
                self.speakfunc("Done boss,please give a file name to save")
                name = self.takeCommand()
                if not os.path.exists("ANSbyASTRA"):
                    os.mkdir("ANSbyASTRA")
                img.save(f"ANSbyASTRA/{name}.png")
                self.speakfunc("Image saved boss")
                self.asktoquit()
            # elif "create a file" in self.query.lower():
            #     if not os.path.exists("data"):
            #         os.mkdir("data")
            #     x= 100
            #     # speakfunc("Enter no of files")
            #     # x = int(input("Enter no of files"))
            #     # speakfunc("ENter file type like png,jpg,pdf,exe,py,c,cpp,doc,etc\n")
            #     # ftype = input("ENter file type like png,jpg,pdf,exe,py,c,cpp,doc,etc\n")
            #     for i in range(x):
            #          os.remove(f'myfile{i + 1}.doc')
            #     print("Sucessfully created")
            #     speakfunc("tero kam va .kam matra garauchas taile fata")

            else:
                self.speakfunc(self.chatfunc(self.query))
                self.asktoquit()


startExecution = MainThread()


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.startTask)
        self.ui.pushButton_2.clicked.connect(self.close)

    def startTask(self):
        self.ui.movie = QtGui.QMovie("VJl.gif")
        self.ui.label_6.setMovie(self.ui.movie)
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("QWfb.gif")
        self.ui.label_7.setMovie(self.ui.movie)
        self.ui.movie.start()
        timer = QTimer(self)
        timer.timeout.connect(self.showtime)
        timer.start(1000)
        startExecution.start()

    def showtime(self):
        timest = QTime.currentTime()
        datest = QDate.currentDate()
        x = timest.toString('hh:mm:ss')
        y = datest.toString(Qt.ISODate)
        scene1 = QGraphicsScene()
        scene2 = QGraphicsScene()
        text_item = QGraphicsTextItem(y)
        text_item2 = QGraphicsTextItem(x)
        scene1.addItem(text_item)
        self.ui.DATE.setScene(scene1)
        scene2.addItem(text_item2)
        self.ui.TIME.setScene(scene2)



        self.ui.CONTENT.setText(f"ASTRA: {chatstr}")
        self.ui.label_5.setText(action)
        # Update the time in the TIME QGraphicsView


app = QApplication(sys.argv)
ASTRA = Main()
ASTRA.show()

exit(app.exec_())
pythoncom.CoUninitialize()
