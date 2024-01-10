import cmd
import subprocess
import sys
import pyttsx3 
import speech_recognition as sr
import datetime
import wikipedia 
import webbrowser
import pywhatkit
import os
import smtplib #for mail
#import openai as ai
# for GUI : 
from PyQt5 import QtWidgets, QtCore,QtGui
from PyQt5.QtCore import QTimer,QTime,QDate, Qt
from PyQt5.QtGui import QMovie
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUiType
from madhavUI import Ui_MadhavUI # from UI file [madhavUI.py] we are importing Ui_MadhavUI class 

###########Importing other python files#######
from difflib import get_close_matches
import json
from random import choices
import normal_chat
from app_control import *
from web_scrapping import *
 


engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate',180)

#function for speak (text to speech)
def speak(audio):
    engine.say(audio)
    engine.runAndWait()

class DateTime:
	def currentTime(self):
		time = datetime.datetime.now()
		x = " A.M."
		if time.hour>12: x = " P.M."
		time = str(time)
		time = time[11:16] + x
		return time

	def currentDate(self):
		now = datetime.datetime.now()
		day = now.strftime('%A')
		date = str(now)[8:10]
		month = now.strftime('%B')
		year = str(now.year)
		result = f'{day}, {date} {month}, {year}'
		return result

#function to wish me 
def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning sir!")
    elif hour>=12 and hour<18:
        speak("Good Afternoon sir!")   
    else:
        speak("Good Evening sir!")  

dictdata = json.load(open('assets/dict_data.json', encoding='utf-8'))
def getMeaning(word):
	if word in dictdata:
		return word, dictdata[word]
	elif len(get_close_matches(word, dictdata.keys())) > 0:
		word = get_close_matches(word, dictdata.keys())[0]
		return word, dictdata[word], 0
	else:
		return word, ["This word doesn't exists in the dictionary."], -1



class MainThread(QThread):
    def __init__(self):
        super(MainThread,self).__init__()

    #function to take commands from me(work as ears)
    def takeCommand(self):
        #It takes microphone input from the user and returns output
        r = sr.Recognizer()
        with sr.Microphone() as source:
            madhavAI.terminalPrint("Jarvis: Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)
        try:
            madhavAI.terminalPrint("Jarvis: Recognizing sir...")
            query = r.recognize_google(audio, language='en-in')
            madhavAI.terminalPrint(f"you said: {query}\n")
        except Exception as e:
            madhavAI.terminalPrint("Jarvis: Say that again sir...") 
            return "None"
        return query
    
    def TaskExecution(self):
         speak("Anything else sir") 

    

    def run(self):
        speak("Your personal Assistent program started. Waiting for your command.")
        while True:
            self.query = self.takeCommand()
            
            if 'morning' in self.query or 'evening' in self.query or 'noon' in self.query:
                wishMe()

            if 'tell me about' in self.query:
                speak('Searching Wikipedia about this sir')
                self.query = self.query.replace("wikipedia", "")
                results = wikipedia.summary(self.query, sentences=2)
		speak("According to Wikipedia")
		madhavAi.terminalPrint(results)
		speak(results)
            
            if 'meaning' in self.query or 'dictionary' in self.query or'definition' in self.query or 'define' in self.query:
                speak('Specify the word to be searched.')
                madhavAI.terminalPrint('Specify the word to be searched.')
                wd =  self.takeCommand()
                meaning_result = getMeaning(wd)
                madhavAI.terminalPrint(meaning_result)
                speak("Meaning of " + str(wd) + " is: " + str(meaning_result))

            if 'battery' in self.query or  'system info' in self.query:
                result = OSHandler(self.query)
                if len(result)==2:
                    speak(result[0])
                    madhavAI.terminalPrint(result[1])
                else:
                    speak(result)
            
            if 'volume' in self.query:
                volumeControl(self.query)		
                speak('Volume Settings Changed')
            
            if 'screenshot' in self.query:
                Win_Opt(self.query)
                madhavAI.terminalPrint("Screen Shot taken")
                speak("Screen Shot Taken")

            
            if 'open Google' in self.query:
                speak('What do you want to search on google')
                madhavAI.terminalPrint('What do you want to search on google')
                cm =  self.takeCommand()
                webbrowser.open(f"{cm}")
                speak("Openning google and searching"+f"{cm}" +"sir please wait..")
            

            if 'open YouTube' in self.query:
                speak('What do you want to play on Youtube')
                madhavAI.terminalPrint('What do you want to paly on Youtube')
                cm =  self.takeCommand()
                Youresult= "https://www.youtube.com/results?search_query=" + cm
                webbrowser.open(Youresult)
                pywhatkit.playonyt(Youresult)
                speak("Openning Youtube and playing"+f"{cm}" +"sir please wait..")
                madhavAI.terminalPrint("Openning Youtube and playing"+f"{cm}" +"sir please wait..")
                # self.query = self.query.replace("")
                # Youresult= "https://www.youtube.com/results?search_query=" + self.query
                # webbrowser.open(Youresult)
                # pywhatkit.playonyt(Youresult)
                # madhavAI.terminalPrint("Opening Youtube sir please wait..")
                
            
            if 'open my chrome' in self.query:
                codePath= "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
                os.startfile(codePath)

            if 'play music' in self.query:
                music_dir = "C:\\Users\\Uttam\\Music"
                songs = os.listdir(music_dir)
                speak("Playing Songs sir please wait..")
                print(songs)    
                os.startfile(os.path.join(music_dir, songs[5]))

            if 'type' in self.query or'save' in self.query or 'delete'in self.query or'select'in self.query or'enter' in self.query:
                System_Opt(self.query)

            if 'open Word' in self.query or 'open Notepad' in self.query or 'open calculator' in self.query or 'open Paint' in self.query:
                System_Opt(self.query)

            if 'window' in self.query or 'close that' in self.query:
                Win_Opt(self.query)
            
            if 'tab' in self.query:
                Tab_Opt(self.query)
            
            


            if 'date' or 'time' in self.query:
                speak(normal_chat.chat(self.query))
                madhavAI.terminalPrint(normal_chat.chat(self.query))

    #working on myfiles
            # if 'open my data science folder' in self.query:
            #     codePath= "D:\\Data science"
            #     os.startfile(codePath)

            # if 'open my movies' in self.query:
            #     codePath = "E:\\Movies"
            #     os.startfile(codePath)
            #     speak("Opening your movies folder sir..")
            

    #closing program
            if 'close the program' in self.query:
                speak("Closing program sir,..see you afterwards.")
                codePath= "MadhavAI.py"
                os.closefile(codePath)

		
            
                


startExecution = MainThread()

class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MadhavUI()  # calling GUI
        self.ui.setupUi(self) 
        self.ui.pushButton.clicked.connect(self.startTask) #will change to voice command which will activate madhav program.
        self.ui.pushButton_2.clicked.connect(self.close)

    def startTask(self):
        self.ui.movie = QtGui.QMovie("D:\\My Projects\\MadhavAI\\UIdesigns\\QjoV.gif")
        self.ui.label_2.setMovie(self.ui.movie)
        self.ui.movie.start() 
        self.ui.movie = QtGui.QMovie("D:\\My Projects\\MadhavAI\\UIdesigns\\7LP8.gif")
        self.ui.label_3.setMovie(self.ui.movie) 
        self.ui.movie.start()
        self.ui.movie = QtGui.QMovie("D:\\My Projects\\MadhavAI\\UIdesigns\\Jarvis_Loading_Screen.gif")
        self.ui.label_4.setMovie(self.ui.movie) 
        self.ui.movie.start()
        timer = QTimer(self)
        timer.timeout.connect(self.showTime)
        timer.start(1000)
        #self.ui.textBrowser_3.append('>>')
        #self.ui.textBrowser_3.setText(MainThread.__init__(self))
        #self.ui.textBrowser_3.setText(MainThread.TaskExecution(self))
        startExecution.start()
        

    def showTime(self):
        current_time = QTime.currentTime()
        current_date = QDate.currentDate()
        label_time = current_time.toString('hh:mm:ss')
        label_date = current_date.toString(Qt.ISODate)
        self.ui.textBrowser.setText(label_date)
        self.ui.textBrowser_2.setText(label_time)

    def terminalPrint(self, text):
        if isinstance(text, tuple):
            text = ' '.join(str(item) for item in text)
        self.ui.terminalOutputBox.appendPlainText(text)



if __name__ =='__main__':
    app = QApplication(sys.argv)
    madhavAI = Main()
    madhavAI.show()
    exit(app.exec_())








