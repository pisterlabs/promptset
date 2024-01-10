# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:30:44 2021

@author: pyfbsdk59@gmail.com / https://github.com/pyfbsdk59/YTMockingjay
"""

import tkinter as tk
#from tkinter.ttk import Separator
from tkinter import messagebox


def YTDownload(YTurl, xfilename):
    import yt_dlp

    # audio settings
    ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': xfilename,
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    }


    # construct yt_dlp object
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([YTurl])

def WhipserTranscriber(xfilename, TOKEN):
    import os
    import openai
    openai.api_key = TOKEN
    audio_file = open(xfilename + ".mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    transcript_text = transcript.to_dict().get('text')

    print(transcript_text)
    
    return transcript_text
    
def ChatGPTTranscriber(transcript_text, TOKEN, system_prompt):
    import os
    import openai
    openai.api_key = TOKEN
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": transcript_text}
    ]
    )
    
    #"請你成為文章的小幫手，將以下文章加上中文標點符號並且適切地斷行和分段，以繁體中文輸出"
    
    print("以下為文章整理：")
    print(completion.choices[0].message['content'].strip())
    
##########################################################################################
class App1(tk.PanedWindow):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        
        
        self.lbTitle = tk.Label(self, text= "YTMockingjay 影片擷取聲音檔轉成文字並加標點符號", font="Helvetic 15 bold")
        self.lbTitle.pack() 



        self.variable_YTurl = tk.StringVar(self)
        self.lb_YTurl = tk.Label(self, text= "1. YT URL")
        self.entry_YTurl = tk.Entry(self, textvariable=self.variable_YTurl) 

        self.lb_YTurl.pack()
        self.entry_YTurl.pack(pady=15)



        self.variable_xfilename = tk.StringVar(self)
        self.lb_xfilename = tk.Label(self, text= "2. 儲存檔案名稱（不須副檔名）")
        self.entry_xfilename = tk.Entry(self, textvariable=self.variable_xfilename) 

        self.lb_xfilename.pack()
        self.entry_xfilename.pack(pady=10)
        


        self.button1 = tk.Button(self)
        self.button1["text"] = "開始執行轉換"
        self.button1["command"] = self.YT2textMain  #這裡的 YT2textMain不使用()，也不帶入參數
        self.button1.pack(pady=20)



    
        
        
        


    def YT2textMain(self):  #排程程式主體

    
        import csv
        settings_list = []
        with open('settings.csv', 'r', newline='', encoding='utf-8-sig') as csvfile:
            rows = csv.reader(csvfile)
                #一行只有一個代號
            for row in rows:
                settings_list.append(row[0])

        TOKEN = settings_list[0]
        system_prompt = settings_list[1] #不要斷行 要在同一行





        def YT2textBatch():  

            
            YTurl = self.variable_YTurl.get()
            xfilename = self.variable_xfilename.get()



            YTDownload(YTurl, xfilename)
            
            import time
            time.sleep(5)

            transcript_text2 = WhipserTranscriber(xfilename, TOKEN)

            ChatGPTTranscriber(transcript_text2, TOKEN, system_prompt)

            
            
        
        YT2textBatch()        


class App101(tk.PanedWindow):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        #self.pack(side='left')
        #self.configure(background='lavender')

        #self.sep2 = Separator(self, orient="horizontal")
        #self.sep2.pack(fill="x", pady=20)
        

        self.button10 = tk.Button(self)
        self.button10["text"] = "Z-1 顯示工作資料夾（下載檔案所在處）"
        self.button10["command"] = self.ShowGetCWD 
        self.button10.pack(pady=10)


     

        self.button13 = tk.Button(self)
        self.button13["text"] = "Z-2 聯絡開發者"
        self.button13["command"] = self.ContactCoder
        self.button13.pack(pady=10) 
        
        self.sep1 = Separator(self, orient="horizontal")
        self.sep1.pack(fill="x", pady=20)        



    def ContactCoder(self):
        messagebox.showinfo("聯絡開發者", "姓名: Nick Hwang, 電子郵件: pyfbsdk59@gmail.com, Github: https://github.com/pyfbsdk59/YTMockingjay")



    def ShowGetCWD(self):  #顯示工作資料夾，也就是執行檔所在的資料夾


        #import numpy as np
        import os
        #import shutil
        
        path1 = str(os.path.abspath(os.getcwd()))  #工作資料夾     
        messagebox.showinfo("工作資料夾dist：", path1)
       
####################
############################
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.ttk import Separator
from tkinter import messagebox
from tkinter import *


root = tk.Tk()
root.geometry("650x500")  
root.title("YTMockingjay YT2text轉換程式 Beta 3")
root.resizable(1, 1)


notebook = ttk.Notebook(root) 
#frame0 = App(root)  #frame0

#frame101 = App101(notebook) #frame1
#frame101.pack(side='top')
#sep2 = Separator(root, orient="horizontal")
#sep2.pack(fill="x", pady=20)
frame1 = App1(notebook) #frame1
frame1.pack(side='top')

frame101 = App101(notebook) #frame1
frame101.pack(side='top')

notebook.add(frame1,text="A. YTMockingjay YT2text轉換程式")   # 建立頁次1同時插入Frame1
notebook.add(frame101,text="Z. 聯絡作者")   # 建立頁次3同時插入Frame3

notebook.pack(padx=10,pady=10,fill="both",expand=True)    
#panes = tk.Tk().PanedWindow()

root.mainloop()
