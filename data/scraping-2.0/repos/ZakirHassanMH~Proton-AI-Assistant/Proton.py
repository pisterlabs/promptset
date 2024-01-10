import openai
import pyttsx3
from tkinter import messagebox
from tkinter import *
import tkinter.font as tkFont
import urllib.request
import speech_recognition as sr
import requests
from bs4 import BeautifulSoup
import random
from googletrans import Translator
from gtts import gTTS
import os
import playsound

root = Tk()
root.title("Proton")
root.geometry('955x600')
root.resizable(False, False)
root.configure(bg='#0A2647')

openai_api=input('Enter the API key: ')

photo = PhotoImage(file = "icon.ico")
root.iconphoto(False, photo)

fontObj = tkFont.Font(size=28)
fontObj1 = tkFont.Font(size=20)
fontObj2 = tkFont.Font(size=15)

User = Text(root, state='disabled', width=61, height=2,font=fontObj1,bg='#205295',fg='#F2F7A1')
User.place(x=16,y=125)

AI = Text(root, state='disabled', width=61, height=10,font=fontObj1,bg='#205295',fg='#F2F7A1')
AI.place(x=16,y=250)

translator = Translator()

def connect(host='http://google.com'):
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False
connection="connected" if connect() else "no"

options = [
    "English",
    "Tamil",
    "Sinhala"
]
clicked = StringVar()
clicked.set( "English" )
drop = OptionMenu( root , clicked , *options)
drop.config(font=fontObj2,bg='#A5C9CA')
drop.place(x=840,y=16.5)


input = Entry(root, font=fontObj, width=33,bg='#205295',fg='#00FFF6')
input.grid(column=0, row=0, columnspan=5, padx=15, pady=15)
text1=Label(root,text='Proton: ',font=fontObj1,bg='#0A2647',fg='#F2F7A1')
text1.place(x=15,y=205)
text2=Label(root,text='User: ',font=fontObj1,bg='#0A2647',fg='#F2F7A1')
text2.place(x=15,y=80)

r = sr.Recognizer()
m = sr.Microphone()

def voice():
  global Lang
  Lang=clicked.get()
  
  if Lang=='English':
    try:
        with m as source: audio = r.listen(source)
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google(audio)
            # we need some special handling here to correctly print unicode characters to standard output
            if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                Fin=value.lower()
                User.configure(state='normal')
                User.delete('1.0','end')
                User.insert('end',Fin)
                User.configure(state='disabled')
                AI.configure(state='disabled')
                if connection=='no':
                    messagebox.showerror("ERROR", "Connect the internet")
                elif 'date'in Fin:
                    user_query = Fin
                
                    URL = "https://www.google.co.in/search?q=" + user_query
                
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
                
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')
                
                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
                elif 'president' in Fin or 'prime minister' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='FLP8od').get_text()
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')
                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait() 
                elif 'time' in Fin:
                    user_query = Fin

                    URL = "https://www.google.co.in/search?q=" + user_query

                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }

                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='vk_gy vk_sh card-section sL6Rbf').get_text()

                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
            
                elif Fin=='':
                    messagebox.showwarning("Warning", "Enter the input")
                elif 'your name' in Fin or 'hello' in Fin or 'hi' in Fin or 'hey'in Fin:
                    hello = ["Hello! I'm Proton. How can I help you?", "Hi, I'm Proton. What's you name?", "I'm Proton. What's up?","I'm Proton. What should I do for you?"]
                            
                    out=random.choice(hello)
                elif 'bye' in Fin or Fin=='close':
                    root.destroy()
            
                elif 'thanks' in Fin or'thank you' in Fin or'thankyou'in Fin:
                    out="Ok, It's my pleasure"
                else:
            
                    openai.api_key = openai_api

                    response = openai.Completion.create(
                      model="text-davinci-003",
                      prompt=Fin,
                      temperature=0.9,
                      max_tokens=1000,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0.6,
                      stop=[" Human:", " AI:"]
                    )
                    out = response['choices'][0]['text']

                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(out)
                    engine.runAndWait()
            else:
                Fin=value.lower()
                User.configure(state='normal')
                User.delete('1.0','end')
                User.insert('end',Fin)
                User.configure(state='disabled')
                AI.configure(state='disabled')
                if connection=='no':
                    messagebox.showerror("ERROR", "Connect the internet")
                elif 'date'in Fin:
                    user_query = Fin
                
                    URL = "https://www.google.co.in/search?q=" + user_query
                
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
                
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')
                
                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
                elif 'president' in Fin or 'prime minister' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='FLP8od').get_text()
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')
                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
                elif 'time' in Fin:
                    user_query = Fin

                    URL = "https://www.google.co.in/search?q=" + user_query

                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }

                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='vk_gy vk_sh card-section sL6Rbf').get_text()

                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
                elif 'weather' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='wob_dcp').get_text()
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',result)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(result)
                    engine.runAndWait()
            
                elif Fin=='':
                    messagebox.showwarning("Warning", "Enter the input")
                elif 'your name' in Fin or 'hello' in Fin or 'hi' in Fin or 'hey'in Fin:
                    hello = ["Hello! I'm Proton. How can I help you?", "Hi, I'm Proton. What's you name?", "I'm Proton. What's up?","I'm Proton. What should I do for you?"]
                            
                    out=random.choice(hello)
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(out)
                    engine.runAndWait()
                elif 'bye' in Fin or Fin=='close':
                    root.destroy()
            
                elif 'thanks' in Fin or'thank you' in Fin or'thankyou'in Fin:
                    out="Ok, It's my pleasure"
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(out)
                    engine.runAndWait()
                else:
            
                    openai.api_key = openai_api

                    response = openai.Completion.create(
                      model="text-davinci-003",
                      prompt=Fin,
                      temperature=0.9,
                      max_tokens=1000,
                      top_p=1,
                      frequency_penalty=0,
                      presence_penalty=0.6,
                      stop=[" Human:", " AI:"]
                    )
                    out = response['choices'][0]['text']

                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')

                    engine.setProperty('rate', 150)
                    engine.say(out)
                    engine.runAndWait()
                
        except sr.UnknownValueError:
           messagebox.showinfo("Information", "Oops! Didn't catch that") 
    except:
        pass    
  elif Lang=='Tamil':
    try:
        with m as source: audio = r.listen(source)
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google (audio, language="ta-IN")
            # we need some special handling here to correctly print unicode characters to standard out
            end=str(value)
            User.configure(state='normal')
            User.delete('1.0','end')
            User.insert('end',end)
            User.configure(state='disabled')
            AI.configure(state='disabled')
            EngText=translator.translate(text=end, dest='en')
            Fin=EngText.text
            
            if connection=='no':
                messagebox.showerror("ERROR", "Connect the internet")
            elif 'date'in Fin:
                user_query = Fin
            
                URL = "https://www.google.co.in/search?q=" + user_query
            
                headers = {
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                }
                page = requests.get(URL, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                result = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
                TamilText=translator.translate(text=result, dest='ta')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='ta')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            elif 'president' in Fin or 'prime minister' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='FLP8od').get_text()
                    TamilText=translator.translate(text=result, dest='ta')
                    out=TamilText.text

                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')
                    tts = gTTS(text=out, lang='ta')
                    filename = "abc.mp3"
                    tts.save(filename)
                    playsound.playsound(filename)
                    os.remove(filename)
            elif 'time' in Fin:
                user_query = Fin
                URL = "https://www.google.co.in/search?q=" + user_query
                headers = {
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                }
                page = requests.get(URL, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                result = soup.find(class_='vk_gy vk_sh card-section sL6Rbf').get_text()
                TamilText=translator.translate(text=result, dest='ta')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='ta')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
        
            elif Fin=='':
                messagebox.showwarning("Warning", "Enter the input")
            elif 'your name' in Fin or 'hello' in Fin or 'hi' in Fin or 'hey'in Fin:
                hello = ["Hello! I'm Proton. How can I help you?", "Hi, I'm Proton. What's you name?", "I'm Proton. What's up?","I'm Proton. What should I do for you?"]
                        
                result=random.choice(hello)
                TamilText=translator.translate(text=result, dest='ta')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='ta')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            elif 'weather' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='wob_dcp').get_text()
                    TamilText=translator.translate(text=result, dest='ta')
                    out=TamilText.text
                
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')
                    tts = gTTS(text=out, lang='ta')
                    filename = "abc.mp3"
                    tts.save(filename)
                    playsound.playsound(filename)
                    os.remove(filename)
            elif 'bye' in Fin or Fin=='close':
                root.destroy()
        
            elif 'thanks' in Fin or'thank you' in Fin or'thankyou'in Fin:
                result="Ok, It's my pleasure"
                TamilText=translator.translate(text=result, dest='ta')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='ta')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            else:
        
                openai.api_key = openai_api
                response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=Fin,
                  temperature=0.9,
                  max_tokens=1000,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0.6,
                  stop=[" Human:", " AI:"]
                )
                result = response['choices'][0]['text']
                TamilText=translator.translate(text=result, dest='ta')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='ta')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
        except sr.UnknownValueError:
           messagebox.showinfo("Information", "Oops! Didn't catch that") 
    except:
        pass 
  elif Lang=='Sinhala':
    try:
        with m as source: audio = r.listen(source)
        try:
            # recognize speech using Google Speech Recognition
            value = r.recognize_google (audio, language="si")
            # we need some special handling here to correctly print unicode characters to standard out
            end=str(value)
            User.configure(state='normal')
            User.delete('1.0','end')
            User.insert('end',end)
            User.configure(state='disabled')
            AI.configure(state='disabled')
            EngText=translator.translate(text=end, dest='en')
            Fin=EngText.text
                
            if connection=='no':
                messagebox.showerror("ERROR", "Connect the internet")
            elif 'date'in Fin:
                user_query = Fin
            
                URL = "https://www.google.co.in/search?q=" + user_query
            
                headers = {
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                }
                page = requests.get(URL, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                result = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
                TamilText=translator.translate(text=result, dest='si')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='si')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            elif 'president' in Fin or 'prime minister' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='FLP8od').get_text()
                    TamilText=translator.translate(text=result, dest='si')
                    out=TamilText.text
                
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')
                    tts = gTTS(text=out, lang='si')
                    filename = "abc.mp3"
                    tts.save(filename)
                    playsound.playsound(filename)
                    os.remove(filename)
            elif 'weather' in Fin:
                    user_query = Fin
                    URL = "https://www.google.co.in/search?q=" + user_query
                    headers = {
                    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                    }
                    page = requests.get(URL, headers=headers)
                    soup = BeautifulSoup(page.content, 'html.parser')
                    result = soup.find(class_='wob_dcp').get_text()
                    TamilText=translator.translate(text=result, dest='si')
                    out=TamilText.text
                
                    AI.configure(state='normal')
                    AI.delete('1.0',END)
                    AI.insert('end',out)
                    AI.configure(state='disabled')
                    tts = gTTS(text=out, lang='si')
                    filename = "abc.mp3"
                    tts.save(filename)
                    playsound.playsound(filename)
                    os.remove(filename)
            elif 'time' in Fin:
                user_query = Fin
                URL = "https://www.google.co.in/search?q=" + user_query
                headers = {
                'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
                }
                page = requests.get(URL, headers=headers)
                soup = BeautifulSoup(page.content, 'html.parser')
                result = soup.find(class_='vk_gy vk_sh card-section sL6Rbf').get_text()
                TamilText=translator.translate(text=result, dest='si')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='si')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
        
            elif Fin=='':
                messagebox.showwarning("Warning", "Enter the input")
            elif 'your name' in Fin or 'hello' in Fin or 'hi' in Fin or 'hey'in Fin:
                hello = ["Hello! I'm Proton. How can I help you?", "Hi, I'm Proton. What's you name?", "I'm Proton. What's up?","I'm Proton. What should I do for you?"]
                        
                result=random.choice(hello)
                TamilText=translator.translate(text=result, dest='si')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='si')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            elif Fin=='bye'or Fin=='close':
                root.destroy()
        
            elif 'thanks' in Fin or'thank you' in Fin or'thankyou'in Fin:
                result="Ok, It's my pleasure"
                TamilText=translator.translate(text=result, dest='si')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='si')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
            else:
        
                openai.api_key = openai_api
                response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=Fin,
                  temperature=0.9,
                  max_tokens=1000,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0.6,
                  stop=[" Human:", " AI:"]
                )
                result = response['choices'][0]['text']
                TamilText=translator.translate(text=result, dest='si')
                out=TamilText.text
            
                AI.configure(state='normal')
                AI.delete('1.0',END)
                AI.insert('end',out)
                AI.configure(state='disabled')
                tts = gTTS(text=out, lang='si')
                filename = "abc.mp3"
                tts.save(filename)
                playsound.playsound(filename)
                os.remove(filename)
        except sr.UnknownValueError:
           messagebox.showinfo("Information", "Oops! Didn't catch that") 
    except:
        pass 
def click():
  

  txt = input.get()
  input.delete(0,END)
  global Fin
  Fin = txt.lower()
  User.configure(state='normal')
  User.delete('1.0','end')


  User.insert('end',Fin)
  User.configure(state='disabled')
  AI.configure(state='disabled')
  if connection=='no':
      messagebox.showerror("ERROR", "Connect the internet")
  elif 'date'in Fin:
      user_query = Fin
  
      URL = "https://www.google.co.in/search?q=" + user_query
  
      headers = {
      'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
      }
      page = requests.get(URL, headers=headers)
      soup = BeautifulSoup(page.content, 'html.parser')
      result = soup.find(class_='vk_bk dDoNo FzvWSb').get_text()
  
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',result)
      AI.configure(state='disabled')
  
      engine.setProperty('rate', 150)
      engine.say(result)
      engine.runAndWait()
  elif 'time' in Fin:
      user_query = Fin
      URL = "https://www.google.co.in/search?q=" + user_query
      headers = {
      'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
      }
      page = requests.get(URL, headers=headers)
      soup = BeautifulSoup(page.content, 'html.parser')
      result = soup.find(class_='vk_gy vk_sh card-section sL6Rbf').get_text()
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',result)
      AI.configure(state='disabled')
      engine.setProperty('rate', 150)
      engine.say(result)
      engine.runAndWait()  
  elif 'weather' in Fin:
    user_query = Fin
    URL = "https://www.google.co.in/search?q=" + user_query
    headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
    }
    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    result = soup.find(class_='wob_dcp').get_text()
    AI.configure(state='normal')
    AI.delete('1.0',END)
    AI.insert('end',result)
    AI.configure(state='disabled')
    engine.setProperty('rate', 150)
    engine.say(result)
    engine.runAndWait()
  elif 'president' in Fin or 'prime minister' in Fin:
      user_query = Fin
      URL = "https://www.google.co.in/search?q=" + user_query
      headers = {
      'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36 Edg/89.0.774.57'
      }
      page = requests.get(URL, headers=headers)
      soup = BeautifulSoup(page.content, 'html.parser')
      result = soup.find(class_='FLP8od').get_text()
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',result)
      AI.configure(state='disabled')
      engine.setProperty('rate', 150)
      engine.say(result)
      engine.runAndWait()  
  elif Fin=='':
      messagebox.showwarning("Warning", "Enter the input")
  elif 'your name' in Fin or 'hello' in Fin or 'hi' in Fin or 'hey'in Fin:
      hello = ["Hello! I'm Proton. How can I help you?", "Hi, I'm Proton. What's you name?", "I'm Proton. What's up?","I'm Proton. What should I do for you?"]
              
      out=random.choice(hello)
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',out)
      AI.configure(state='disabled')
      engine.setProperty('rate', 150)
      engine.say(out)
      engine.runAndWait()
  elif 'bye' in Fin or Fin=='close':
      root.destroy()  
  elif 'thanks' in Fin or'thank you' in Fin or'thankyou'in Fin:
      out="Ok, It's my pleasure"
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',out)
      AI.configure(state='disabled')
      engine.setProperty('rate', 150)
      engine.say(out)
      engine.runAndWait()
  else:  
      openai.api_key = openai_api
      response = openai.Completion.create(
        model="text-davinci-003",
        prompt=Fin,
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
      )
      out = response['choices'][0]['text']
      AI.configure(state='normal')
      AI.delete('1.0',END)
      AI.insert('end',out)
      AI.configure(state='disabled')
      engine.setProperty('rate', 150)
      engine.say(out)
      engine.runAndWait()



photoimage = PhotoImage(file = "1.png")
send_button = Button(root, image=photoimage, command=click,bg='#A5C9CA')
send_button.grid(column=10, row=0)

photoimage1 = PhotoImage(file = "2.png")
send_button2 = Button(root, image=photoimage1,bg='#A5C9CA',command=voice)
send_button2.place(x=785,y=13)

engine = pyttsx3.init()

root.mainloop()
