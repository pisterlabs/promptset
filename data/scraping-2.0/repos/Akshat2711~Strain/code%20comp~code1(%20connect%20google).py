import requests
from customtkinter import*
from bs4 import BeautifulSoup
from tkinter import*
from PIL import ImageTk,Image
import mysql.connector as mycon
from datetime import date
import time
from gtts import gTTS
import openai
from apikey import *
import os
from smtplib import *
import urllib.request
import speech_recognition as sr
import webbrowser
from random import*
language = 'en'#language of output
openai.api_key ="k-YNaaYK1Tx1tsGZoI1YCaT3BlbkFJNxeleKJ3l65fbZbaICQB"


con=mycon.connect(host="localhost",user="root",database="searchhistory",password="27ramome76A")
cur=con.cursor()

set_appearance_mode("dark")#####for appearence of window
set_default_color_theme("green")
#called when result direct on google
def query():
    global result

    user_query = res

    URL = "https://www.google.co.in/search?q=" + user_query

    headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
    }

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    result = soup.find(class_='Z0LcW t2b5Cf').get_text()
    print(result)
    msg1.delete("0.0", "end")
    msg1.insert("0.0",'Ans-  "'+result+'"')
    msg1.place(relx=0.42,rely=0.60)
    but4.place(x=0,y=765)


   

def window1():
    global win1
    global count
    win1=CTk()
    win1.title("main window")
    win1.geometry("1920x1080-10-7")
    count=-1
#list of images used as back in mainwindow 
    list1=[ImageTk.PhotoImage(Image.open("D:\code comp\p1.png")),ImageTk.PhotoImage(Image.open("D:\code comp\p2.png")),ImageTk.PhotoImage(Image.open("D:\code comp\p3.png"))]
    canvas=Canvas(win1,width=5000,height=2000,highlightthickness=0)
    canvas.create_image(0,0,anchor='nw',image=list1[0])
    canvas.pack()
    def next():
        global count
#remeber to change count==2 to no of images in (list-1)
        if count==2:
            canvas.create_image(0,0,anchor='nw',image=list1[0])
            count=0
        else:
            canvas.create_image(0,0,anchor='nw',image=list1[count+1])
            count+=1
        win1.after(5000,next)
    next()
#history being sent  to database
#sent to while loop
    def sgoogle():
        global res
        global date
#date code
        date = date.today()
        res=str(txt1.get())
#current time code
        ###################
        ###############
        t = time.localtime()
        gm=open("getmailforhis.txt","r")
        fm=gm.read()
        gm.close()
        current_time = time.strftime("%H:%M:%S", t)
        cur.execute("insert into history() values('{}','{}','{}','{}');".format(res,date,current_time,fm))
        con.commit()
        whileloop()
    def his():
        win_his=CTk()
        win_his.config()
        win_his.title("History")
        win_his.geometry("1920x1080-10-7")
# to delete history
        def delete_his():
            gm=open("getmailforhis.txt","r")
            fm=gm.read()
            gm.close()
            cur.execute("delete from history where email='{}'".format(fm))
            con.commit()
            lbl_confirm.configure(text="REOPEN TO SEE CHANGES!")
        textbox = CTkTextbox(win_his, height=850, width=760,font=("Helvitica",25))#textbox in which history written   
        but_his=CTkButton(win_his,text="DELETE HISTORY",font=("Helvitica",18),command=delete_his).place(x=695,y=700)
        lbl_confirm=CTkLabel(win_his,text=" ",font=("Helvitica",15))
        lbl_confirm.place(x=620,y=675)
        gm=open("getmailforhis.txt","r")
        fm=gm.read()
        gm.close()
        cur.execute("select * from history where email='{}'".format(fm))
        h=cur.fetchall()
        print(h)

        
        
        for h1 in h:#########################
            textbox.insert("0.0","Query-["+h1[0]+"]   Date-["+h1[1]+"]   Time-["+h1[2]+"]\n")
        textbox.place(x=400, y=10)
        win_his.mainloop()
            
        
#ai fnc(chatgpt 3.5)        
    def ai():
        win1.destroy()
        win_ai=CTk()#mann was here(founder)  
        global countai,list2
        win_ai.title("AI")
        win_ai.geometry("1920x1080-10-7")
        countai=-1
#list of images used as back in aiwindow 
        list2=[ImageTk.PhotoImage(Image.open("D:\code comp\p4.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p5.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p6.jpg"))]
        canvas=Canvas(win_ai,width=5000,height=2000,highlightthickness=0)
        canvas.create_image(0,0,anchor='nw',image=list2[0])
        canvas.pack()
        def next():
            global countai
    #remeber to change count==2 to no of images in (list-1)
            if countai==2:
                canvas.create_image(0,0,anchor='nw',image=list2[0])
                countai=0
            else:
                canvas.create_image(0,0,anchor='nw',image=list2[countai+1])
                countai+=1
            win_ai.after(5000,next)
        next()
        def resai():
            aiCTkEntry=txtai.get()
            dateai = date.today()
            resai=str(aiCTkEntry)
    #current time code
            tai = time.localtime()
            gm=open("getmailforhis.txt","r")
            fm=gm.read()
            gm.close()
            current_timeai = time.strftime("%H:%M:%S", tai)
            cur.execute("insert into history() values('{}','{}','{}','{}');".format(resai+"[AI]",dateai,current_timeai,fm))
            con.commit()
            output = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user","content":str(aiCTkEntry)}])
            msgai.place(x=470, y=80)
            msgai.delete("0.0", "end")
            msgai.insert("0.0",output['choices'][0]['message']['content'])#########################


        def ai_mainback():
            win_ai.destroy()
            window1()
        #create image window
        def createimage():
            global countimg,list3
#after ai CTkButton create image action:
            def aiimg():
                global date
                idea=txtimg.get()
                sizeval=txtimg2.get()
                name=txtimg3.get()
                date = date.today()
        #current time code
                t = time.localtime()
                current_time = time.strftime("%H:%M:%S", t)
                gm=open("getmailforhis.txt","r")
                fm=gm.read()
                gm.close()
                cur.execute("insert into history() values('{}','{}','{}','{}');".format(idea+"[AI(img)]",date,current_time,fm))
                con.commit()
                response = openai.Image.create(prompt=idea,n=1,size=sizeval)
                out=response["data"][0]["url"]


                  
                # Retrieving the resource located at the URL
                # and storing it in the file name a.png
                url = str(out)
                urllib.request.urlretrieve(url,name)
                  
                # Opening the image and displaying it (to confirm its presence)
                img = Image.open(name)
                img.show()
            win_ai.destroy()
            winimage=CTk()
            countimg=-1
    #list of images used as back in aiwindow 
            list3=[ImageTk.PhotoImage(Image.open("D:\code comp\p7.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p8.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p9.jpg"))]
            canvas=Canvas(winimage,width=5000,height=2000,highlightthickness=0)
            canvas.create_image(0,0,anchor='nw',image=list3[0])
            canvas.pack()
            def next():
                global countimg
        #remeber to change count==2 to no of images in (list-1)
                if countimg==2:
                    canvas.create_image(0,0,anchor='nw',image=list3[0])
                    countimg=0
                else:
                    canvas.create_image(0,0,anchor='nw',image=list3[countimg+1])
                    countimg+=1
                winimage.after(5000,next)
            next()
            def aiimgback():
                winimage.destroy()
                window1()
            
            winimage.geometry("1920x1080-10-7")
            winimage.title("CREATE IMAGE")
            txtimg=CTkEntry(winimage,width=630,font=("Helvitica",30))
            txtimg.insert(0, "Enter idea for your image")
            txtimg.place(x=490,y=10)
            txtimg2=CTkComboBox(winimage,font=("Helvitica",30),width=190,values=["1024x1024","256x256","512x512"])
            txtimg2.set("1024x1024")
            txtimg2.place(x=793,y=60)
            txtimg3=CTkEntry(winimage,width=300,font=("Helvitica",30))
            txtimg3.insert(0, "name of image to be created")
            txtimg3.place(x=490,y=60)
            butimg=CTkButton(winimage,text="CREATE",font=("Helvitica",30),command=aiimg,width=10).place(x=986,y=60)
            butimgback=CTkButton(winimage,text="HOME",font=("Helvitica",20),command=aiimgback).place(x=0,y=764)
            
            winimage.mainloop()
#speech to text code
        def speech_txt():
            r=sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                print("say something")
                audio=r.listen(source)
#time sleep to recognize voice
                time.sleep(2)
                try:
                    inputaudio=r.recognize_google(audio)
                    print("you have said:\n"+inputaudio)
                    output = openai.ChatCompletion.create(model="gpt-3.5-turbo",messages=[{"role": "user","content":str(inputaudio)}])
                    msgai.place(x=470, y=80)
                    msgai.delete("0.0", "end")
                    msgai.insert("0.0",output['choices'][0]['message']['content'])
                    ############################################
                    myobj = gTTS(text=output['choices'][0]['message']['content'], lang=language, slow=False)
                    myobj.save("micout.mp3")
                    os.system("micout.mp3")
                    dateaimic = date.today()
                    gm=open("getmailforhis.txt","r")
                    fm=gm.read()
                    gm.close()
#current time code for mic ai
                    tai = time.localtime()
                    current_timeaim = time.strftime("%H:%M:%S", tai)
                    cur.execute("insert into history() values('{}','{}','{}','{}');".format(inputaudio+"[AI(mic)]",dateaimic,current_timeaim,fm))
                    con.commit()


                except Exception as e:
                    print("error")
                    msgai.delete("0.0", "end")
                    msgai.insert("0.0", "TRY AGAIN!")
                
                    
                
            
        txtai=CTkEntry(win_ai,width=900,font=("Helvitica",19))
        txtai.place(x=350,y=10)
        lblai1=CTkLabel(win_ai,text="STRAIN",font=("Helvitica",19)).place(x=1468,y=0)
        butai=CTkButton(win_ai,text="FIND",font=("Helvitica",19),command=resai,width=10).place(x=1251,y=10)
        butai2=CTkButton(win_ai,text="BACK",font=("Helvitica",19),command=ai_mainback).place(x=0,y=766)
        msgai=CTkTextbox(win_ai,width=700,height=600,font=("Helvitica",20))
        butai3=CTkButton(win_ai,text="CREATE IMAGE USING AI",font=("Helvitica",19),command=createimage).place(x=1290,y=766)
        butai4=CTkButton(win_ai,command=speech_txt,text="MIC",font=("Helvitica",19),width=10).place(x=300,y=10)
        win_ai.mainloop()
    global msg1
#to open google visible when query 2 fnc runs
    def opengoogle():
        q=txt1.get()
        webbrowser.open('https://google.com/'+'search?q='+q)
    def logout():
        win1.destroy()
        login()
    global but4
    txt1=CTkEntry(win1,width=970,font=("Helvitica",30))
    txt1.place(x=220,y=350)
    but1=CTkButton(win1,text="search",font=("Helvitica",30),command=sgoogle,width=15).place(x=1190,y=350)
    lbl1=CTkLabel(win1,text="STR",font=("Helvitica",60)).place(x=605,y=200)
    lbl2=CTkLabel(win1,text="N",font=("Helvitica",60)).place(x=787,y=200)
    but2=CTkButton(win1,text="History",font=("Helvitica",18),command=his,width=10).place(x=1469,y=0)
    but3=CTkButton(win1,text="AI",font=("Helvitica",54),width=10,command=ai).place(x=723,y=200)
    but4=CTkButton(win1,font=("Helvitica",14),text="OPEN IN BROWSER",command=opengoogle)
    msg1=CTkTextbox(win1,font=("Helvitica",20))#####################
    but5=CTkButton(win1,font=("Helvitica",14),text="LOG OUT",width=10,command=logout).place(x=1457,y=765)
    #to diplay welcome message
    f=open("himsgfile(code1).txt","r")
    namehello=f.read()
    f.close()
    lbl3=CTkLabel(win1,text="Welcome "+namehello,font=("Helvitica",20)).place(x=0,y=0)
    win1.mainloop()
#called when result no direct on google
def query2(res2):
    global result
    user_query = res2

    URL = "https://www.google.co.in/search?q=" + user_query

    headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36'
    }

    page = requests.get(URL, headers=headers)
    soup = BeautifulSoup(page.content, 'html.parser')
    result = soup.find(class_='VwiC3b yXK7lf MUxGbd yDYNvb lyLwlc lEBKkf').get_text()
    print(result)
    msg1.delete("0.0", "end")
    msg1.insert("0.0",'Ans-  "'+result+'"')
    but4.place(x=0,y=765)
    msg1.place(relx=0.42,rely=0.60)
#deciding whether result directly on google or not
def whileloop():
    while True:
        try:
            query()
        except Exception:
            query2(res)
        break

def login():
    loginwin=CTk()
    global count_login
    count_login=-1
    #list of images used as back in aiwindow 
    list4=[ImageTk.PhotoImage(Image.open("D:\code comp\p10.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p11.jpg")),ImageTk.PhotoImage(Image.open("D:\code comp\p12.jpg"))]
    canvaslogin=Canvas(loginwin,width=5000,height=2000,highlightthickness=0)
    canvaslogin.create_image(0,0,anchor='nw',image=list4[0])
    canvaslogin.pack()
##############
    def next():
        global count_login
        #remeber to change count==2 to no of images in (list-1)
        if count_login==2:
            canvaslogin.create_image(0,0,anchor='nw',image=list4[0])
            count_login=0
        else:
            canvaslogin.create_image(0,0,anchor='nw',image=list4[count_login+1])
            count_login+=1
        loginwin.after(9000,next)
    next()
    loginwin.title("LOGIN")
    loginwin.geometry("1920x1080-10-7")
    def back_sign_login():
        loginwin.destroy()
        login()
    def signupcmd():
         signinbut.place(x=10000,y=700)###code to remove CTkButton from screen
         signupbut.place(x=10000,y=650)####code to remove CTkButton from screen
         namelogin.place(x=570,y=350)
         gmaillogin.place(x=570,y=400)
         passlogin.place(x=570,y=450)
         otplogin.place(x=570,y=500)
         contbut2.place(x=1307,y=762)
         backbtn.place(x=0,y=762)
         otpbtn.place(x=975,y=348)
        
            
    def signincmd():
         otplogin.place(x=3000,y=500)
         contbut2.place(x=3000,y=700)
         namelogin.place(x=3000,y=350)
         lbl1_login.place(x=3000,y=350)
         otpbtn.place(x=3000,y=100)
         ####################above 6 line code just to remove CTkButton and CTkLabels out of screen
         gmaillogin.place(x=570,y=400)
         passlogin.place(x=570,y=450)
         signinbut.place(x=10000,y=700)
         signupbut.place(x=10000,y=650)
         contbut.place(x=1397,y=762)
         backbtn.place(x=0,y=762)
    
    def getotp():
        global rando
        gm=str(gmaillogin.get())
        na=str(namelogin.get())
        rando=randint(1000,10000)
        send="Hi "+na+",\n"+"OTP for your STRAIN browser is  "+str(rando)
        s_e="cs.pr0j3ct.xii@gmail.com"#sender email
        passwd="omtghmrwfehjgcqb"#pass of sender
        server=SMTP("smtp.gmail.com",587)
        server.starttls()
        server.login(s_e,passwd)
        server.sendmail(s_e,gm,send)
       
        
    def cont_to_signin():
        o=otplogin.get()
        ####################################CTkLabel for account created?
        if str(rando)==o:
            pa=str(passlogin.get())
            na=str(namelogin.get())
            gm=str(gmaillogin.get())
            
            cur.execute("select* from logininf;")
            allinf=cur.fetchall()
            for i in allinf:
                if i[1]==gm:
                
                    lbl1_login.configure(text="account already exist!")
                    lbl1_login.place(x=650,y=290)
                else:
                    cur.execute("insert into logininf() values('{}','{}','{}');".format(na,gm,pa))
                    con.commit()
                    signincmd()
                            
            
        else:
            lbl1_login.configure(text="otp incorrect")
            lbl1_login.place(x=650,y=290)
        
        
    
        
    def cont_to_main():
        g=gmaillogin.get()
        p=passlogin.get()
        cur.execute("select* from logininf;")
        allinf=cur.fetchall()
        for i in allinf:
            if (g,p)==(i[1],i[2]):
                gm=open("getmailforhis.txt","w")
                gm.write(str(g))
                gm.close()
                f=open("himsgfile(code1).txt","w")
                f.write(i[0])
                f.close()
                loginwin.destroy()
                window1()
                break
            else:
                lbl1_login.configure(text="TRY AGAIN!")
                lbl1_login.place(x=650,y=305)
        
        
    #login image
    signinbut=CTkButton(loginwin,text="SIGN IN",font=("Helvitica",20),command=signincmd)
    signinbut.place(x=690,y=700)
    signupbut=CTkButton(loginwin,text="SIGN UP",font=("Helvitica",19),command=signupcmd)
    signupbut.place(x=690,y=650)
    gmaillogin=CTkEntry(loginwin,width=400,font=("Helvitica",28))
    gmaillogin.insert(0, "Enter your Email....")
    passlogin=CTkEntry(loginwin,width=400,font=("Helvitica",28))
    passlogin.insert(0, "Enter your Password....")
    namelogin=CTkEntry(loginwin,width=400,font=("Helvitica",28))#remove place on sign up
    namelogin.insert(0, "Enter your Name....")
    otplogin=CTkEntry(loginwin,width=400,font=("Helvitica",28))
    otplogin.insert(0, "Enter your OTP....")
    lbl1_login=CTkLabel(loginwin,text="TRY AGAIN!",font=("Helvitica",27))
    contbut=CTkButton(loginwin,text="CONTINUE",font=("Helvitica",20),command=cont_to_main)
    contbut2=CTkButton(loginwin,text="CONTINUE TO SIGN IN",font=("Helvitica",20),command=cont_to_signin)
    backbtn=CTkButton(loginwin,text="BACK",font=("Helvitica",20),command=back_sign_login)
    otpbtn=CTkButton(loginwin,text="G\nE\nT\n \nO\nT\nP",width=8,font=("Helvitica",23),command=getotp)
    
    
    
    loginwin.mainloop()  
login()
