from customtkinter import *
import tkinter as tk
from PIL import ImageTk,Image
from datetime import *
import time
import pymysql
from tkinter import ttk
import wikipedia
import random
import speech_recognition as sr
import pyttsx3
import tkinter.messagebox as messagebox
import openai
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tkdial import ScrollKnob
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pyautogui
from reportlab.lib import colors
import webbrowser
import socket



student_id=185
flag_bot=True
first_enter=True
#----------------------------------------------------------------Local Connection----------------------------------------------------------------

#database connect
db=pymysql.connect(host='localhost',user='root',password='root',database='student_database',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor) 

#----------------------------------------------------------------Universal Function----------------------------------------------------------------


#flash message for adding form
def flash_message(text, flashlb):
    def clear_flash_message(flashlb):
        flashlb.configure(text="")
        flashlb.update()
    flashlb.configure(text=text)
    flashlb.update()
    flashlb.after(3000, clear_flash_message, flashlb)

def reverse_animation( frame_to_animate, speed):
    target_width = frame_to_animate.cget("width")
    target_height = frame_to_animate.cget("height")
    current_width = target_width
    current_height = target_height

    while current_width > 0 or current_height > 0:
        current_width = max(current_width - speed, 0)
        current_height = max(current_height - speed, 0)
        frame_to_animate.configure(width=current_width, height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)

def animate_frame(frame_to_animate):
    target_height = frame_to_animate.cget("height")
    frame_to_animate.configure(height=0)
    
    current_height = 0

    while current_height < target_height:
        current_height = min(current_height + 30, target_height)
        frame_to_animate.configure( height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)

def animate_text( element_to_animate, speed):
    sentence = element_to_animate.cget("text")
    element_to_animate.configure(text="")
    element_to_animate.update()

    def animate(index):
        if index < len(sentence):
            element_to_animate.configure(text=element_to_animate.cget("text") + sentence[index])
            element_to_animate.update()
            element_to_animate.after(speed, animate, index + 1)

    animate(0)

def animate_bot_frame( frame_to_animate, speed):
    target_width = frame_to_animate.cget("width")
    target_height = frame_to_animate.cget("height")
    frame_to_animate.configure(width=0, height=0)
    current_width = 0
    current_height = 0

    while current_width < target_width or current_height < target_height:
        current_width = min(current_width + speed, target_width)
        current_height = min(current_height + speed, target_height)
        frame_to_animate.configure(width=current_width, height=current_height)
        frame_to_animate.update()
        #time.sleep(0.01)

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    
    engine.say(text)
    engine.runAndWait()


#----------------------------------------------------------------Additional Data Access----------------------------------------------------------------


#data access from database

cursor=db.cursor()
query1=f"Select * from student where stud_id={student_id}"
cursor.execute(query1)
s_data=cursor.fetchall()
data1=s_data[0]
su_id=data1.get("stud_id")
Name=data1.get("name")
Dob=data1.get("dob")
Std=data1.get("std_code")
Address=data1.get("address")
mobile=data1.get("phone_no")
gender=data1.get("gen")

cursor = db.cursor()
cursor.execute("select count(*) from student where std_code=%s and stud_id<%s", (Std, student_id))
s_data = cursor.fetchall()
roll_index = s_data[0]["count(*)"]  



#----------------------------------------------------------------Local Function----------------------------------------------------------------


    
#datetime
def date_time_display():
    
    def ct_time():
        now = datetime.now()
        ct_string = now.strftime("%H:%M:%S")
        return ct_string

    def ct_change():
        ct_string = ct_time()
        time_lb.configure(text=ct_string)
        f0.after(1000, ct_change)  #update every 1 second
    #logout_frame
    def logout_frame():
        delete_frames()
        date_time_display()
        def destroy_window():
            student_win.destroy()
            #login_page()
        #asking to logout
        log_lb=CTkLabel(f0,text="Do you want to logout ?",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
        log_lb.place(relx=0.4,y=200,anchor=CENTER)
        log_bu=CTkButton(f0,text="Yes",height=45,command=destroy_window,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        log_bu.place(relx=0.4,y=300,anchor=CENTER)

    today = datetime.today()
    t_date= today.strftime("%B %d, %Y")
    #date and time 
    d_f=CTkFrame(f0,width=350,height=50,border_color="black",border_width=3,fg_color="white",corner_radius=40)
    d_f.place(x=730,y=5)
    time_lb=CTkLabel(d_f,width=110,height=30,text="",font=CTkFont("Helvetica",19),fg_color="white",corner_radius=40,text_color="black")
    time_lb.place(relx=0.8,rely=0.5,anchor=CENTER)
    date_lb=CTkLabel(d_f,text=t_date,width=150,height=30,corner_radius=8,font=("Helvetica",20),text_color="black",bg_color="white")
    date_lb.place(relx=0.3,rely=0.5,anchor=CENTER)
    ct_change()

    photo1=CTkImage(Image.open("logout1.png"),size=(50,50))
    edit_b=CTkButton(f0,image=photo1,command=logout_frame,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#66B3FF",corner_radius=10)
    edit_b.place(x=1100,y=0)


#to delete frames
def delete_frames():
    global flag_bot 
    if flag_bot==False:
        reverse_animation(frame1,30)
        flag_bot=True
    f0.update()
    
    for f in f0.winfo_children():
        f.destroy()
    f0.update()

#to indicate
def indicate(lb,frame):
    hide_indicators()
    lb.configure(fg_color="#0066ff")
    delete_frames()
    frame()

#hide indicators
def hide_indicators():
    home_indicate.configure(fg_color="white")
    search_indicate.configure(fg_color="white")
    schedule_indicate.configure(fg_color="white")
    complain_indicate.configure(fg_color="white")
    attendance_indicate.configure(fg_color="white")


def display_connection_status():
    global first_enter,internet_bot
    def connect_check():
        IPaddress = socket.gethostbyname(socket.gethostname())
        if IPaddress == "127.0.0.1":
            return False
        else:
            return True 
    network_stat = connect_check()
    def dis():
        connect_check_lb.configure(fg_color="#66B3FF")
    def update_status():
        connect_display()
        connect_check_lb.after(3000, update_status)
    def connect_display():
        nonlocal network_stat
        network_Checked = connect_check()
        
        if network_Checked == True and network_stat == False:
            connect_check_lb.configure(fg_color="green")
            animate_bot_frame(connect_check_lb,5)
            flash_message("Internet Connected",connect_check_lb)
            connect_check_lb.after(3000,dis)
            network_stat = True
        if network_Checked == False and network_stat == True:
            connect_check_lb.configure(fg_color="red")
            animate_bot_frame(connect_check_lb,5)
            flash_message("No internet",connect_check_lb)
            connect_check_lb.after(3000,dis)
            network_stat = False

    connect_check_lb=CTkLabel(f0,text="",height=45,width=150,corner_radius=20,text_color="white",font=CTkFont("Helvetica",20,"bold"))
    connect_check_lb.place(relx=0.4,rely=0.95,anchor=CENTER)
    if first_enter:   
        if network_stat == True:
            connect_check_lb.configure(fg_color="green")
            animate_bot_frame(connect_check_lb,5)
            flash_message("Internet Connected",connect_check_lb)
            connect_check_lb.after(3000,dis)
        else:
            connect_check_lb.configure(fg_color="red")
            animate_bot_frame(connect_check_lb,5)
            flash_message("No internet",connect_check_lb)
            connect_check_lb.after(3000,dis)
        first_enter=False

    connect_check_lb.after(1000, update_status)



#--------------------------------------------home frames-----------------------------------------------------------------------------------



#home_frame
def home_frame(): 
    #date and time 
    date_time_display()
    l2=CTkLabel(f0,text=("Welcome "+ Name),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
    l2.place(x=40,y=0)
    #animate_text(l2,20)
    
    f0.after(1000,display_connection_status)
    
    #identiy card
    f7=CTkFrame(f0,width=550,height=338,fg_color="#ffffe6",border_width=3,corner_radius=12,border_color="black")
    f7.place(x=330,y=150,anchor="n")
    photo5=CTkImage(Image.open("id_back.png"),size=(538,160))
    l3=CTkLabel(f7,image=photo5,text="")
    l3.place(x=6,y=5)
    photo6=CTkImage(Image.open("background.jpeg"),size=(538,18))
    l3=CTkLabel(f7,image=photo6,text="")
    l3.place(x=6,y=333,anchor="sw")

    photo6=CTkImage(Image.open("student_male.png"),size=(150,150))
    if gender=='M':
        pass
    else:
        photo6=CTkImage(Image.open("student_female.png"),size=(150,150))

    l4=CTkLabel(f7,image=photo6,text=" ")
    l4.place(x=315,y=90)

    photo7=CTkImage(Image.open("logo.png"),size=(70,70))
    l5=CTkLabel(f7,image=photo7,text=" ",bg_color='#6FD0FE')
    l5.place(x=60,y=15)

    l6=CTkLabel(f7,text="IDENTITY CARD",bg_color='#6FD0FE',font=("Helvetica",25),text_color="black")
    l6.place(x=170,y=33)

    id_lb=CTkLabel(f7,text=("STU.ID :  " + str(su_id)),width=100,height=30,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
    id_lb.place(x=15,y=113)

    name_lb=CTkLabel(f7,text=("NAME :   " + Name),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
    name_lb.place(x=15,y=140)

    dob_lb=CTkLabel(f7,text=("DOB    :  " + str(Dob)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
    dob_lb.place(x=15,y=175)

    std_lb=CTkLabel(f7,text=("STD    :  "+ Std),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
    std_lb.place(x=15,y=210)

    mobile_lb=CTkLabel(f7,text=("MOBILE NO. :  " + str(mobile)),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6")
    mobile_lb.place(x=15,y=245)

    address_lb=CTkLabel(f7,text=("ADDRESS :  " + Address),width=100,height=35,corner_radius=8,font=("Helvetica",15),text_color="black",bg_color="#ffffe6",wraplength=500)
    address_lb.place(x=15,y=280)
    messages = [{'role': 'system', 'content': f'behave like a bot named as Edubot owned by Edutrack ; which is develoved by manoj,dhruv and ankit ; sycs students at vivekanand education society ;dont use EduBot: text in your responce; you will be use by a student namely {Name},whose student id is {su_id} , he or shes address is {Address},mobile no. is {mobile} and studies in {Std}'}]
    def EduBot():
        
        global row_count,frame1 

        openai.api_key = '<Openai token key>'
        def AI_Responce(user_message):
            messages.append({"role": "user", "content": user_message})
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=100
            )
            reply = response["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": reply})
            return reply

        row_count =0

        

        def print_message(text1):
            global row_count
            chat1 = CTkLabel(chat_frame, text=text1, height=20, wraplength=300, corner_radius=12, text_color="black", fg_color="#ccffe6", font=CTkFont("Helvetica", 15))
            chat1.grid(column=1,row=row_count,sticky="e", pady=(13,10), padx=(5, 10),ipady=7)

            photo1=CTkImage(Image.open("student_male.png"),size=(30,30))
            if gender=='M':
                pass
            else:
                photo1=CTkImage(Image.open("student_female.png"),size=(30,30))
            l1=CTkLabel(chat_frame,image=photo1,text=" ")
            l1.grid(column=2,row=row_count,sticky="e")

            row_count+=1
            student_win.update()
            bot_reply =AI_Responce(text1)
            print_reply(bot_reply,(2,2))

        def print_reply(text1,valu):
            global chat2,row_count
            time.sleep(0)
            photo2=CTkImage(Image.open("chat_bot.png"),size=(30,30))
            l2 = CTkButton(chat_frame, image=photo2, command=lambda text2=text1: speak(text2),text="", width=0.1, hover_color="#e0e0e7", cursor="hand2", fg_color="#ECF2FF")
            l2.grid(column=0, row=row_count, sticky="w", padx=(5, 10))

            chat2 = CTkLabel(chat_frame, text=text1, height=30, wraplength=300, corner_radius=12, text_color="black", fg_color="#AEE2FF", font=CTkFont("Helvetica", 15))
            chat2.grid(column=1,row=row_count,sticky="w", padx=(0, 5),pady=valu,ipady=7)
            animate_text(chat2, 25)
            
            row_count+=1

        

        def send():
            request = entry1.get()
            entry1.delete(0, tk.END)
            if len(request) != 0:
                print_message(request)

        frame1 = CTkFrame(f0, width=550, height=590,fg_color="#FFD0D0",corner_radius=30)
        frame1.place(x=330,y=100,anchor="n")
        animate_bot_frame(frame1,10)
        
        lab2=CTkLabel(frame1,text="bot",width=80,height=80,fg_color="#FFD0D0",corner_radius=80,font=CTkFont("Helvetica", 12))
        lab2.place(relx=0.5,rely=0,anchor=CENTER)
        photo1=CTkImage(Image.open("chat_bot.png"),size=(45,45))
        Bot_But=CTkButton(lab2,image=photo1,command=Bot_Button,text="",width=50,hover_color="#FFD0D0",cursor="hand2",fg_color="#FFD0D0",corner_radius=400)
        Bot_But.place(relx=0.5,rely=0.5,anchor=CENTER)

        chat_frame = CTkScrollableFrame(frame1, height=500, width=500,fg_color="#ECF2FF",scrollbar_button_color="#ECF2FF")
        chat_frame.place(relx=0.5, rely=0.47, anchor=CENTER)

        chat_frame.columnconfigure(0, weight=1)
        chat_frame.columnconfigure(1, weight=15)
        chat_frame.columnconfigure(2, weight=1)
        entry1 = CTkEntry(frame1,placeholder_text="Say hello to EduBot",width=450,height=35)
        entry1.place(relx=0.02, rely=0.95,anchor="w")
        send_button = CTkButton(frame1, text="Send",width=70,height=35, command=send)
        send_button.place(relx=0.98, rely=0.95, anchor="e")
        student_win.bind('<Return>', lambda event=None: send_button.invoke())
        
        frame1.update()
        frist=AI_Responce("greet us in 10 words;my name is"+Name)
        print_reply(frist,(20,0))

    def Bot_Button():
        global flag_bot 
        if flag_bot==True:
            EduBot()
            flag_bot=False
        else:
            reverse_animation(frame1,30)
            flag_bot=True
    
    lab2=CTkLabel(f0,text="bot",width=80,height=80,fg_color="#FFD0D0",corner_radius=80,font=CTkFont("Helvetica", 12))
    lab2.place(x=330,y=100,anchor=CENTER)
    photo1=CTkImage(Image.open("chat_bot.png"),size=(45,45))
    Bot_But=CTkButton(lab2,image=photo1,command=Bot_Button,text="",width=50,hover_color="#FFD0D0",cursor="hand2",fg_color="#FFD0D0",corner_radius=400)
    Bot_But.place(relx=0.5,rely=0.5,anchor=CENTER)

    
    #attendance_grap
    def attendance_graph(num_rows):
        def hide(btn):
            btn_day.configure(fg_color="#33CCFF")
            btn_week.configure(fg_color="#33CCFF")
            btn_month.configure(fg_color="#33CCFF")
            btn.configure(fg_color="#888888")
        if num_rows == 0:
            hide(btn_day)
        elif num_rows == 6:
            hide(btn_week)
        elif num_rows == 29:
            hide(btn_month)

        f8 = CTkFrame(f0, width=550, height=480, fg_color="#ccb3ff", border_width=3, corner_radius=12, border_color="black")
        f8.place(x=630, y=100)
        #animate_frame(f8)
        cursor.execute("select * from attendance WHERE attendance.attendance_date >= DATE(NOW()) - INTERVAL %s DAY and stand_code=%s order by attendance_date;", (num_rows,Std,))
        b_data = cursor.fetchall()

        subject_lec = [0, 0, 0, 0, 0, 0, 0]
        tot_attendance = [0, 0, 0, 0, 0, 0, 0]

        for i in range(len(b_data)):
            data1 = b_data[i]
            sub = data1["sub_code"]
            attendance_list = list(data1["attend_sheet"])

            if sub == 1:
                subject_lec[0] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[0] += 1
            elif sub == 2:
                subject_lec[1] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[1] += 1
            elif sub == 3:
                subject_lec[2] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[2] += 1
            elif sub == 4:
                subject_lec[3] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[3] += 1
            elif sub == 5:
                subject_lec[4] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[4] += 1
            elif sub == 6:
                subject_lec[5] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[5] += 1
            elif sub == 7:
                subject_lec[6] += 1
                if attendance_list[roll_index] == "1":
                    tot_attendance[6] += 1


        tot_subject = ['Eng', 'Hin', 'Mar', 'Math', 'Sci', 'Social', 'Pt']
        x = np.arange(len(tot_subject))
        fig = plt.figure(figsize=(5.40, 4.7))
        fig.set_facecolor("#e0e0eb")

        sns.lineplot(x=x, y=subject_lec, marker='o', label="Total_Lecture")
        sns.lineplot(x=x, y=tot_attendance, marker='o', label="Lecture_Attended")

        plt.xlabel('SUBJECT', fontstyle='italic', fontsize=12)
        plt.ylabel('ATTENDANCE(%)', fontstyle='italic', fontsize=12)
        plt.title('YOUR ATTENDANCE', fontweight='bold', fontsize=20)
        plt.xticks(x, tot_subject)
        plt.ylim([0, max(subject_lec) + 2])
        plt.legend()
        canvas = FigureCanvasTkAgg(fig, master=f8)
        canvas.draw()
        canvas.get_tk_widget().place(x=5, y=5)


    btn_day = CTkButton(f0, text=" Day ", command=lambda: attendance_graph(0), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
    btn_day.place(x=760, y=620,anchor=CENTER)

    btn_week = CTkButton(f0, text=" Week ", command=lambda: attendance_graph(6), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
    btn_week.place(x=925, y=620,anchor=CENTER)

    btn_month = CTkButton(f0, text=" Month ", command=lambda: attendance_graph(29), hover_color="#D9D9D0", height=37, width=125, border_width=2, corner_radius=20, border_color="black", text_color="black", fg_color="#33CCFF", font=CTkFont("Helvetica", 20))
    btn_month.place(x=1080, y=620,anchor=CENTER)
    btn_day.invoke()

    #quotes
    def quotes():
        try:
            # list of quotes
            quotes = [
                "Be the change you wish to see in the world.",
                "The only way to do great work is to love what you do.",
                "Not all who wander are lost.",
                "Believe you can and you're halfway there.",
                "If you want to go fast, go alone. If you want to go far, go together.",
                "It does not matter how slowly you go as long as you do not stop.",
                "Life is what happens when you're busy making other plans.",
                "It is during our darkest moments that we must focus to see the light.",
                "You must be the change you wish to see in the world.",
                "The future belongs to those who believe in the beauty of their dreams.",
                "The best way to predict the future is to invent it."]

            # randomly select a quote from the list
            selected_quote = random.choice(quotes)

            f6=CTkFrame(f0,width=550,height=100,fg_color="#ffe6cc",border_width=3,corner_radius=12,border_color="black")
            f6.place(x=55,y=520)
            quotes_lb=CTkLabel(f6,text="Thought of the day!",width=50,height=20,font=("bold",30),text_color="black",bg_color="transparent")
            quotes_lb.place(relx=0.5,rely=0.3,anchor=CENTER)
            label=CTkLabel(f6,text=selected_quote ,width=40,height=20,font=("bold",18),text_color="black",bg_color="transparent",wraplength=510)
            label.place(relx=0.5,rely=0.7,anchor=CENTER)
            animate_text(label,25)
            quotes_wall=CTkImage(Image.open("qoutes_bg.png"),size=(70,50))
            quotes_wlb=CTkLabel(f6,text="",image=quotes_wall,width=50,height=20,bg_color="transparent")
            quotes_wlb.place(relx=0.1,rely=0.3,anchor=CENTER)
        except Exception as e:
            pass
    quotes()



#------------------------------------------------------search_frame-----------------------------------------------------------------------------------------------

def search_frame():
    #date and time 
    date_time_display()
    #talk
    def talk():
        try:  
            speak(summary)
        except:
            speak("There is nothing to read, please search something first")

    def search():
        global result,summary
        try:
            result = input.get()
            if result=="":
                raise speak("Please search something in search Bar !")
            summary = wikipedia.summary(result, sentences=3)
            text.configure(state="normal")
            text.delete('1.0',END) 
            text.insert('1.0',summary)
            text.configure(state="disable")
        except wikipedia.exceptions.WikipediaException:
            speak(f"Could not request results for ,{result}")

    
    def listen():
        input.delete(0, END)
        input.configure(placeholder_text=" Search with StudentHub")
        messagebox.showinfo("Status", "Tap to Speak...")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
                
        try:
            result = recognizer.recognize_google(audio)
            input.delete(0, END)
            input.insert(0, result)
            speak("You said, " + result)
            f0.update()
            search()
        except wikipedia.exceptions.WikipediaException:
            speak(f"Could not request results for ,{result}")
        except:
            speak("Could not understand audio")
        


    
    #student section
    l2=CTkLabel(f0,text="Search with StudentHub",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
    l2.place(x=40,y=0)
    #animate_text(l2,20)
    input = CTkEntry(f0, width = 350,height=40,font=("halvetica",20),fg_color="#ffe6cc",text_color="black",border_width=3,corner_radius=12,border_color="black",placeholder_text=" Search with StudentHub",placeholder_text_color="grey")
    input.place(x=100,y=90)

    text = CTkTextbox(f0,border_width=3,border_color="black",width=1040,corner_radius=15,height=531,font=("halvetica",20),fg_color="#ffe6cc")
    text.place(relx=0.44,rely=0.58,anchor=CENTER)
    text.configure(state="disable")
    animate_frame(text)

    photo7=CTkImage(Image.open("search_logo.png"),size=(30,30))
    button5 = CTkButton(f0,image=photo7,text='',width=50,height=40,fg_color="#e0e0eb")
    button5.configure(command=search)                                                 
    button5.place(x=455,y=90)

    photo8 = CTkImage(Image.open("mic.png"), size=(30, 30))
    button5 = CTkButton(f0, image=photo8, text='', width=50, height=40, fg_color="#e0e0eb",command=listen)
    button5.place(x=515, y=90)

    photo8 = CTkImage(Image.open("speaker.png"), size=(30, 30))
    button5 = CTkButton(f0, image=photo8, text='', width=50, height=40, fg_color="#e0e0eb",command=talk)
    button5.place(x=575, y=90)

    

#-------------------------------------------------------------academic_frame-----------------------------------------------------------------------


#Academic_frame
def academic_frame():
    global taught_bt,grade_bt

    #Treeview
    def treeview():
        f91=CTkFrame(f0,width=900,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        f91.place(x=300,y=100)
        animate_frame(f91)
        cursor=db.cursor()
        cursor.execute("SELECT teacher.teacher_id,teacher.name,teacher.quali ,teacher.phone_no,subject.sub_name FROM teacher, class , teach_class , subject where teacher_id=teacher_code and sub_id=sub_code and std_id=std_code and std_id=%s",(Std))
        data=cursor.fetchall()
        teach_id=[]
        teach_name=[]
        teach_quali=[]
        subject_name=[]

        for i in data:
                teacher_id=i.get("teacher_id")
                teacher_name=i.get("name")
                teacher_quali=i.get("quali")
                sub_name=i.get("sub_name")
                
                teach_id.append(teacher_id)
                teach_name.append(teacher_name)
                teach_quali.append(teacher_quali)
                subject_name.append(sub_name)

        stu_table=ttk.Treeview(f91,columns=("teach_id","teach_name","teach_quali","subject_name"),show="headings")
        style=ttk.Style(f91)
    
        style.theme_use("clam")
        style.configure("Treeview",rowheight=50,font=("Roboto"),background="#96DED1",fieldbackground="#96DED1", foreground="black")
        style.configure("Treeview.Heading",font=("Roboto"))
        stu_table.heading("teach_id",text="Teacher_id")
        stu_table.heading("teach_name",text="Teacher_name")
        stu_table.heading("teach_quali",text="Teacher_quali")
        stu_table.heading("subject_name",text="Subject")

        stu_table.column("teach_id",width=170,anchor=CENTER)
        stu_table.column("teach_name",width=250,anchor=CENTER)
        stu_table.column("teach_quali",width=200,anchor=CENTER)
        stu_table.column("subject_name",width=250,anchor=CENTER)

        for i in range(len(teach_name)-1,-1,-1):
                stu_table.insert(parent="",index=0,values=(teach_id[i],teach_name[i],teach_quali[i],subject_name[i]))
        stu_table.place(relx=0.5,rely=0.5,anchor=CENTER)

    def taught_by():
        grade_bt.configure(fg_color="#33CCFF")
        notes_bt.configure(fg_color="#33CCFF")
        taught_bt.configure(fg_color="#888888")
        treeview()
    def grade_frame():         
        def show_grade_stu(exam_type,own_bt,ot_bt):
            global capture_button
            f8=CTkFrame(tabview,width=860,height=480,fg_color="white",bg_color="white",corner_radius=12)
            f8.place(x=20,y=65)
            animate_frame(f8)
            flash_massa=CTkLabel(f8,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20))
            flash_massa.place(x=310,y=440)
            own_bt.configure(fg_color="#888888")
            ot_bt.configure(fg_color= "#33CCFF")
            if exam_type=="Mid Term":
                unit_state="Unit I"
            elif exam_type=="Final Term":
                unit_state="Unit II"
            cursor=db.cursor()
            cursor.execute("select obt_mks from grade where exam_type=%s and stud_code=%s order by(sub_code)",(exam_type,student_id))
            data=cursor.fetchall()
            if len(data)!=7:
                capture_button = CTkButton(tabview, text="",width=60,height=40,corner_radius=20,fg_color= "white",font=CTkFont("Helvetica",20))
                capture_button.place(relx=0.97, y=13,anchor="ne")
                f8.configure(fg_color="white")
                flash_massa.configure(text_color="red")
                flash_message("Grades to be announced",flash_massa)
            else:
                photo6=CTkImage(Image.open("download.png"),size=(30,30))
                capture_button = CTkButton(tabview, text="",image=photo6,width=30,corner_radius=20,hover_color="#e0e0eb",fg_color= "white",font=CTkFont("Helvetica",20), command= lambda :capture_and_save(f8))
                capture_button.place(relx=0.97, y=13,anchor="ne")
                f8.configure(fg_color="#87CEFA")
                sub_lb=CTkLabel(f8,text="Subject",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                sub_lb.place(x=20,y=20)
                unit_lb=CTkLabel(f8,text=unit_state+" /20",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                unit_lb.place(x=190,y=20)
                inter_lb=CTkLabel(f8,text="Internal /20",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                inter_lb.place(x=360,y=20)
                exter_lb=CTkLabel(f8,text="External /80",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                exter_lb.place(x=530,y=20)
                gr_lb=CTkLabel(f8,text="Grades",height=45,width=150,corner_radius=20,text_color="black",fg_color="#1E90FF",font=CTkFont("Helvetica",20))
                gr_lb.place(x=700,y=20)

                cursor=db.cursor()
                cursor.execute("select sub_name from subject order by (sub_id)")
                data=cursor.fetchall()
                new_y=90
                flag_fail=True
                #for subjects
                for i in range(len(data)):
                    subj_lb=CTkLabel(f8,text=data[i]['sub_name'],height=45,width=150,corner_radius=20,text_color="black",font=CTkFont("Helvetica",20))
                    subj_lb.place(x=90,y=new_y,anchor=CENTER)
                    new_y+=50

                #for unit_marks
                cursor=db.cursor()
                cursor.execute("select obt_mks from grade where stud_code=%s and exam_type=%s order by (sub_code)",(student_id,unit_state))
                data=cursor.fetchall()
                new_y=90
                for i in range(len(data)):
                    mk_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    mk_e.place(x=260,y=new_y,anchor=CENTER)
                    mk_e.insert(0,data[i]['obt_mks'])
                    if data[i]['obt_mks']<7:
                        mk_e.configure(fg_color="#ffcccc")
                        flag_fail=False
                    mk_e.configure(state="disabled")
                    new_y+=50

                #for inter_exter_marks
                cursor=db.cursor()
                cursor.execute("select internal_mk,external_mk,obt_mks,obt_grd from grade where stud_code=%s and exam_type=%s order by (sub_code)",(student_id,exam_type))
                term_data=cursor.fetchall()
                new_y=90
                for i in range(len(term_data)):
                    int_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    int_e.place(x=435,y=new_y,anchor=CENTER)
                    ext_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    ext_e.place(x=600,y=new_y,anchor=CENTER)
                    gr_e=CTkEntry(f8,justify="center",corner_radius=20,fg_color="#CCFFCC",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                    gr_e.place(x=770,y=new_y,anchor=CENTER)
                    int_e.insert(0,term_data[i]['internal_mk'])
                    ext_e.insert(0,term_data[i]['external_mk'])
                    gr_e.insert(0,term_data[i]['obt_grd'])
                    if term_data[i]['external_mk']<28:
                        ext_e.configure(fg_color="#ffcccc")
                        flag_fail=False
                    int_e.configure(state="disabled")
                    ext_e.configure(state="disabled")
                    gr_e.configure(state="disabled")
                    new_y+=50
                tot_mk=0
                status="Pass"
                for i in range(len(term_data)):
                    tot_mk+=data[i]['obt_mks']+term_data[i]['obt_mks']
                percent=round(((tot_mk/840)*100),2)
                per_lb=CTkLabel(f8,text="Percentage",height=45,width=150,corner_radius=20,text_color="black",fg_color="#E6E6FA",font=CTkFont("Helvetica",20))
                per_lb.place(x=20,y=420)
                per_e=CTkEntry(f8,justify="center",fg_color="#A2C4E0",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                per_e.insert(0,str(percent)+" %")
                per_e.configure(state="disabled")
                per_e.place(x=180,y=423)
                st_lb=CTkLabel(f8,text="Status",height=45,width=150,corner_radius=20,text_color="black",fg_color="#E6E6FA",font=CTkFont("Helvetica",20))
                st_lb.place(x=530,y=420)
                st_e=CTkEntry(f8,justify="center",fg_color="#A2C4E0",height=40,width=100,text_color="black",font=CTkFont("Helvetica",18))
                st_e.place(x=690,y=423)
                if flag_fail==False:
                    st_e.configure(text_color="red")
                    status="Fail"
                else:
                    st_e.configure(text_color="green")
                st_e.insert(0,status)
                st_e.configure(state="disabled")

        def capture_and_save(tabview):
            screenshot = pyautogui.screenshot(region=(tabview.winfo_rootx(), tabview.winfo_rooty(), tabview.winfo_width(), tabview.winfo_height()))
            screenshot_path = "frame_screenshot.png"
            screenshot.save(screenshot_path)
            screenshots.append(screenshot_path)
            update_pdf()

        def update_pdf():
            pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
            if pdf_path:
                c = canvas.Canvas(pdf_path, pagesize=letter)

                # Add school logo before the school name
                logo_path = "pdf_logo.png"  # Update with your logo file path
                logo_width, logo_height = 100, 100  # Adjust the logo size as needed


                school_name = "EDU TRACK"
                sid="Student Id : "+str(su_id)
                sb_date="DOB : "+str(Dob)
                exam="Mid Term"
                s_stan=" STD : "+str(Std)
                sname="Name : "+Name
                c.setFont("Helvetica-Bold", 30)
                screenshot11=screenshots[0]
                for img_path in screenshot11:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    page_width, page_height = letter

                    # Calculate the scale factor to fit the image within the PDF page
                    scale = min(page_width / img_width, page_height / img_height)

                    scaled_width = img_width * scale
                    scaled_height = img_height * scale
 
                    x = (page_width - scaled_width) / 2
                    y = (page_height - scaled_height) / 2

                    # Draw logo
                    c.drawImage(logo_path, 150, page_height - 100, width=logo_width, height=logo_height)
                    
                    # Draw school name
                    c.drawString(250, page_height - 70, school_name)
                    def drawdetails():
                            c.setFont("Helvetica-Bold", 15)
                            c.drawString(120, page_height - 150, sid)
                            c.drawString(350, page_height - 150, s_stan)
                            c.drawString(120, page_height - 180, sname)
                            c.drawString(350, page_height - 180, sb_date)
                            c.setFillColor(colors.red)
                            c.setFont("Times-Italic", 19)
                            c.drawString(260, page_height - 110, exam)
                    drawdetails()
                    
                    # Draw screenshot
                    c.drawImage(img_path, x, y, width=scaled_width, height=scaled_height)
                    c.showPage()

                c.save()


        grade_bt.configure(fg_color="#888888")
        notes_bt.configure(fg_color="#33CCFF")
        taught_bt.configure(fg_color="#33CCFF") 
        tabview = CTkFrame(f0,width=900,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
        tabview.place(x=300,y=100)
        animate_frame(tabview)
        screenshots = []  
        mid_term=CTkButton(tabview,hover_color="#D9D9D0",text="Mid Term",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        mid_term.configure(command=lambda :show_grade_stu("Mid Term",mid_term,final_term))
        mid_term.place(x=280,y=10) 
        final_term=CTkButton(tabview,hover_color="#D9D9D0",text="Final Term",height=40,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        final_term.configure(command=lambda :show_grade_stu("Final Term",final_term,mid_term))
        final_term.place(x=460,y=10)
        mid_term.invoke() 

    def notes_frame():
        grade_bt.configure(fg_color="#33CCFF")
        taught_bt.configure(fg_color="#33CCFF")
        notes_bt.configure(fg_color="#888888")
        tabview = CTkFrame(f0,width=900,height=560,fg_color="#ffcc99",border_width=3,corner_radius=12,border_color="black")
        tabview.place(x=300,y=100)
        animate_frame(tabview)
        sub_f = CTkFrame(tabview,width=240,height=515,fg_color="#b3d9ff",border_width=3,corner_radius=12,border_color="black")
        sub_f.place(x=30,y=20)
        #fetching classes
        cursor=db.cursor()
        cursor.execute("select sub_name from subject")
        data=cursor.fetchall()
        s_nm=[]
        for i in data:
                std_name=i.get("sub_name")
                s_nm.append(std_name)
        def show_pdf(id):
                cursor=db.cursor()
                cursor.execute('select path from notes where note_id=%s',(id))
                data=cursor.fetchone()
                path = data["path"]
                edge_path="C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
                webbrowser.register('edge', None, webbrowser.BackgroundBrowser(edge_path))
                webbrowser.get('edge').open(path)
                
        def show_notes(btn,sub_name):
            for i in buttons:
                if btn==i:
                    btn.configure(fg_color="#888888")
                else:
                    i.configure(fg_color="#33CCFF")
            scroll_f=CTkScrollableFrame(tabview,width=540,corner_radius=20,border_color='black',border_width=2,fg_color="#ccffcc",height=470)
            scroll_f.place(relx=0.65,rely=0.5,anchor=CENTER)
            cursor=db.cursor()
            cursor.execute("Select note_id,notes.title from notes where notes.sub_code=(select sub_id from subject where sub_name=%s) and std_code=%s",(sub_name,Std))
            data=cursor.fetchall()
            r=0
            y_pad=0
            icon=CTkImage(Image.open("notebook.png"),size=(40,40))
            for i in range(len(data)):
                note_tl=data[i]['title']
                note_id=data[i]['note_id']
                new_b=CTkButton(scroll_f,hover_color="#D9D9D0",command=lambda id=note_id: show_pdf(id),text=note_tl,height=55,width=390,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20),compound='right')
                new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
                animate_text(new_b,25)
                new_l=CTkButton(new_b,image=icon,text='',height=40,command=lambda id=note_id: show_pdf(id),hover_color="#D9D9D0",width=40,text_color="black",fg_color="#33CCFF")
                new_l.place(x=70,rely=0.5,anchor="e")
                r+=1
        r=45
        buttons=[]
        for i in s_nm:
            new_b=CTkButton(sub_f,hover_color="#D9D9D0",text=i,height=45,width=180,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
            new_b.place(x=120,y=r,anchor=CENTER)
            animate_text(new_b,25)
            new_b.configure(command=lambda btn=new_b,param=i: show_notes(btn,param))
            buttons.append(new_b)
            r+=70
        buttons[0].invoke()
    date_time_display()
    l2=CTkLabel(f0,text="Academic information",font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
    l2.place(x=40,y=30)
    f3=CTkFrame(f0,width=240,height=560,fg_color="white",border_width=3,corner_radius=12,border_color="black")
    f3.place(x=40,y=100)
    animate_frame(f3)
    taught_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: taught_by(),text="Taught by",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    taught_bt.place(x=120,y=45,anchor=CENTER)
    animate_text(taught_bt,25)
    grade_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: grade_frame(),text="Grades",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    grade_bt.place(x=120,y=110,anchor=CENTER)
    animate_text(grade_bt,25)
    notes_bt=CTkButton(f3,hover_color="#D9D9D0",command=lambda: notes_frame(),text="Notes",height=45,width=170,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    notes_bt.place(x=120,y=175,anchor=CENTER)
    animate_text(notes_bt,25)
    taught_bt.invoke()


#-------------------------------------------------------------complain_frame----------------------------------------------------------------


#Complain_frame
def complain_frame():
    date_time_display()
    l2=CTkLabel(f0,text=("Complain Section"),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
    l2.place(x=55,y=20)
    #animate_text(l2,20)

    new_f=CTkFrame(f0,width=675,height=585,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
    new_f.place(x=55,y=100)
    animate_frame(new_f)
    sub_lb=CTkLabel(new_f,text="Subject",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    sub_lb.place(x=20,y=20)
    sub_et=CTkEntry(new_f,height=40,width=450,border_width=3,corner_radius=30,font=("Roboto",18))
    sub_et.place(x=200,y=22)

    #list with all teacher's names
    cursor=db.cursor()
    cursor.execute("Select name from teacher")
    data=cursor.fetchall()
    teach_name=[]
    for i in data:
        teacher_name=i.get("name")
        teach_name.append(teacher_name)
    
    to_lb=CTkLabel(new_f,text="To",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    to_lb.place(x=20,y=90)
    to_op=CTkOptionMenu(new_f,height=30,width=190,values=["Admin","All Teachers"]+teach_name)
    to_op.place(x=210,y=95)
    dept_lb=CTkLabel(new_f,text="Department",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    dept_lb.place(x=20,y=155)
    dept_op=CTkOptionMenu(new_f,height=30,width=190,values=["Faculty","Administration","Campus Security","IT Services","Library","Food Services","Maintenance","Student Affairs","Logistics","Sports & Activities","Others"])
    dept_op.place(x=210,y=160)
    desc_lb=CTkLabel(new_f,text="Description",height=45,width=170,corner_radius=20,text_color="black",fg_color="#33CCFF",font=CTkFont("Helvetica",20))
    desc_lb.place(x=20,y=220)
    anon_switch=CTkSwitch(new_f,switch_height=30,switch_width=60,text="Hide",font=CTkFont("Helvetica",20))
    anon_switch.place(x=550,y=240)
    desc_et=CTkTextbox(new_f,corner_radius=10,height=250,width=635,border_width=2,font=("Roboto",18))
    desc_et.place(x=20,y=280)
    id_lb=CTkLabel(new_f,text="Complain id",height=25,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
    id_lb.place(x=15,y=540)
    id_et=CTkEntry(new_f,height=30,width=75,border_width=3,corner_radius=30,font=("Roboto",15))
    id_et.place(x=155,y=540)
    cursor=db.cursor()
    cursor.execute("select max(complain_id)+1 from complain")
    data=cursor.fetchall()
    auto=data[0]
    auto_id=auto.get("max(complain_id)+1")
    id_et.insert(0,auto_id)
    id_et.configure(state="disabled")
    def send():
        flash_massa=CTkLabel(new_f,text_color="green",text="",width=120,height=35,corner_radius=8,font=("Helvetica",20),bg_color="#FFE5B4")
        flash_massa.place(relx=0.5,y=540)
        c_id=id_et.get()
        sub=sub_et.get()
        to=to_op.get()
        depart=dept_op.get()
        anon=anon_switch.get()
        desc=desc_et.get("1.0", "end-1c")
        flag_check=True
        if len(sub)==0 or len(desc)==0:
            flag_check=False
            flash_massa.configure(text_color="red")
            flash_message("Try Again",flash_massa)
        if flag_check==True:
            cursor=db.cursor()
            cursor.execute("insert into complain (complain_id, stud_code, subject, `to`, depart, description,hide) values(%s,%s,%s,%s,%s,%s,%s)",(c_id,student_id,sub,to,depart,desc,anon))
            db.commit()
            all_complains()
            #wipping the textbox
            id_et.configure(state="normal")
            id_et.delete(0,END)
            id_et.insert(0,auto_id+1)
            id_et.configure(state="disabled")
            sub_et.delete(0,END)
            desc_et.delete(1.0,END)
            flash_massa.configure(text_color="green")
            flash_message("Sent",flash_massa)
            
    send_b=CTkButton(new_f,command=send,text="Send",height=35,width=100,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
    send_b.place(x=540,y=540)

    #All complains
    new_f=CTkFrame(f0,width=455,height=585,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
    new_f.place(x=750,y=100)
    animate_frame(new_f)
    def solution(comp_id):
        cursor=db.cursor()
        cursor.execute("Select description,solution from complain where complain_id=%s",(comp_id))
        data=cursor.fetchall()
        desc=data[0]["description"]
        sol=data[0]["solution"]
        desc_f=CTkFrame(new_f,width=422,height=490,fg_color="#FFE5B4",border_width=0,corner_radius=12)
        desc_f.place(relx=0.5,rely=0.55,anchor=CENTER)
        sol_t=CTkTextbox(desc_f,font=CTkFont("Helvetica",20),width=400,height=230,fg_color="#e6f7ff",border_width=3,corner_radius=12,border_color="black")
        sol_t.place(x=10,y=250)
        desc_t=CTkTextbox(desc_f,font=CTkFont("Helvetica",20),width=400,height=230,fg_color="#ffff99",border_width=3,corner_radius=12,border_color="black")
        desc_t.place(x=10,y=10)
        if sol==None:
            desc_t.delete('0.0','end')
            desc_t.insert('0.0',desc)
            desc_t.configure(state="disabled")
            sol_t.delete('0.0','end')
            sol_t.insert('0.0',"Action in Process")
            sol_t.configure(state="disabled")
        else: 
            desc_t.delete('0.0','end')
            desc_t.insert('0.0',desc)
            desc_t.configure(state="disabled")
            sol_t.delete('0.0','end')
            sol_t.insert('0.0',sol)
            sol_t.configure(state="disabled")
        all=CTkLabel(new_f,text=" ",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#FFE5B4",font=CTkFont("Helvetica",20))
        all.place(x=20,y=20)
        #back button
        photo1=CTkImage(Image.open("back.png"),size=(40,40))
        edit_b=CTkButton(new_f,command=all_complains,image=photo1,text="",hover_color="#E0E0EB",cursor="hand2",width=20,height=40,fg_color="#FFE5B4",corner_radius=10)
        edit_b.place(x=20,y=18)
        sol_lb=CTkLabel(new_f,text="Solution",height=45,width=170,corner_radius=20,text_color="black",fg_color="#e4d1d1",font=CTkFont("Helvetica",20))
        sol_lb.place(x=250,y=20)  
    def all_complains():
        all=CTkLabel(new_f,text="All Complains",height=45,width=170,corner_radius=20,text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
        all.place(x=20,y=20) 
        sol_lb=CTkLabel(new_f,text=" ",height=45,width=170,corner_radius=20,text_color="black",fg_color="#FFE5B4",font=CTkFont("Helvetica",20))
        sol_lb.place(x=250,y=20)
        scroll_f=CTkScrollableFrame(new_f,width=380,corner_radius=20,fg_color="#B6E5D8",height=450)
        scroll_f.place(relx=0.5,rely=0.55,anchor=CENTER)
        cursor=db.cursor()
        cursor.execute("Select complain_id,depart from complain where stud_code=%s",(student_id))
        data=cursor.fetchall()
        r=0
        y_pad=0
        for i in range(len(data)):
            sub=data[i]['complain_id']
            sub=str(sub)
            dep=data[i]['depart']
            new_b=CTkButton(scroll_f,command=lambda param=sub: solution(param),hover_color="#D9D9D0",text=sub+"     "+dep,height=45,width=330,border_width=2,corner_radius=20,border_color="black",text_color="black",fg_color= "#33CCFF",font=CTkFont("Helvetica",20))
            new_b.grid(row=r,column=1,padx=10,pady=y_pad+10)
            r+=1
    all_complains()



#------------------------------------------------------Attendance Frame---------------------------------------------------------
def attendance_frame():
    date_time_display()
    l2=CTkLabel(f0,text=("Detailed Attendance"),font=CTkFont(family="Helvetica",weight="bold",size=50),text_color="black")
    l2.place(x=55,y=20)
    new_f=CTkFrame(f0,width=1040,height=531,fg_color="#FFE5B4",border_width=3,corner_radius=12,border_color="black")
    new_f.place(relx=0.44,rely=0.58,anchor=CENTER)
    animate_frame(new_f)
    cursor.execute("SELECT attend_sheet, sub_code FROM attendance WHERE stand_code = %s ORDER BY attendance_date DESC", (Std,))
    b_data = cursor.fetchall()

    subject_lec = [0, 0, 0, 0, 0, 0, 0]
    tot_attendance = [0, 0, 0, 0, 0, 0, 0]

    for i in range(len(b_data)):
        data1 = b_data[i]
        sub = data1["sub_code"]
        attendance_list = list(data1["attend_sheet"])
    

        if sub == 1:
            subject_lec[0] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[0] += 1
        elif sub == 2:
            subject_lec[1] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[1] += 1
        elif sub == 3:
            subject_lec[2] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[2] += 1
        elif sub == 4:
            subject_lec[3] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[3] += 1
        elif sub == 5:
            subject_lec[4] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[4] += 1
        elif sub == 6:
            subject_lec[5] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[5] += 1
        elif sub == 7:
            subject_lec[6] += 1
            if attendance_list[roll_index] == "1":
                tot_attendance[6] += 1

    
    
    tot_subject = ['  English', '   Hindi', ' Marathi', '   Maths', ' Science', 'Social sci', 'phy.training']

    def arrange_graphs_horizontally():
        num_graphs = len(tot_subject)
        cols = 4
        row_height = 220
        col_width = 200
        x_padding = 20
        y_padding = 20
        x_space = 60  
        y_space = 40  

        for i in range(num_graphs):
            row = i // cols
            col = i % cols
            attend_per = int((tot_attendance[i] / subject_lec[i]) * 100)
            x_offset = (col * (col_width + x_space)) + x_padding
            y_offset = (row * (row_height + y_space)) + y_padding
            if i>3:
                frame_subjects = CTkFrame(new_f, height=row_height, width=col_width, corner_radius=15, fg_color="white", bg_color="#FFE5B4")
                frame_subjects.place(x=x_offset +130, y=y_offset)
                animate_bot_frame(frame_subjects,15)
            else:
                frame_subjects = CTkFrame(new_f, height=row_height, width=col_width, corner_radius=15, fg_color="white", bg_color="#FFE5B4")
                frame_subjects.place(x=x_offset, y=y_offset)
                animate_bot_frame(frame_subjects,15)
            if attend_per > 75:
                progress_color = "green"
            elif attend_per > 50 and attend_per <= 75:
                progress_color = "yellow"
            else:
                progress_color = "red"

            if attend_per >25 and attend_per <50:
                end=100
            elif attend_per <=50:
                end=180
            elif attend_per<=60 and attend_per >50:
                end=200
            elif attend_per<=75 and attend_per >60:
                end=300
            elif attend_per<=80 and attend_per >75:
                end=400
            elif attend_per<=90 and attend_per >80:
                end=500
            elif attend_per>90 and attend_per<95:
                end=750

            
            if attend_per == 100:
                knob = ScrollKnob(master=frame_subjects, text="", steps=10, radius=110, bar_color="#fffcfc",outer_color="green", 
                                  outer_length=10,border_width=30, start_angle=90, inner_width=0,outer_width=20,text_font="calibri 1", 
                                  text_color="black", fg="#fffcfc", end=100, progress_color="white", start=0)
            else:
                knob = ScrollKnob(master=frame_subjects, text="", steps=10, radius=160, bar_color="#fffcfc",outer_color="#fffcfc", 
                                  outer_length=10,border_width=30, start_angle=90, inner_width=0, outer_width=2,text_font="calibri 1", 
                                  text_color="black", fg="#fffcfc", end=end, progress_color=progress_color, start=0)
            knob.place(relx=0.5,rely=0.4, anchor="center") 

            knob.set(attend_per)
            sub_label = CTkLabel(knob, text=("  "+str(attend_per) + "%" + "\n" + str(tot_subject[i])))
            sub_label.place(relx=0.5,rely=0.5,anchor=CENTER)  
            sep_label=CTkLabel(frame_subjects, text="- - - - - - - - - - - - - - -",font=("Arial Bold", 15))
            sep_label.place(x=40, y=row_height - 65)
            hello_label = CTkLabel(frame_subjects, text=(" Total    : " + str(subject_lec[i]) + " " + "\n Present : " + str(tot_attendance[i]) + " "),font=("Arial Bold", 15), text_color="black")
            hello_label.place(x=45, y=row_height - 40)  


    arrange_graphs_horizontally()

#------------------------------------------------------main window code ----------------------------------------------------------------------

#main window
def main_window():
    global f0,student_id,home_indicate,search_indicate,schedule_indicate,complain_indicate,student_win,attendance_indicate

    set_appearance_mode("light")
    set_default_color_theme("blue")
    student_win=CTk()
    
    student_win.title("Student home page")
    screen_width = student_win.winfo_screenwidth()
    screen_height= student_win.winfo_screenheight()
    student_win_width = screen_width
    student_win_height = screen_height
    student_win.geometry(f"{student_win_width}x{student_win_height}")
    student_win.geometry("+0+0")
    student_win.maxsize(width=1400,height=750)
    student_win.minsize(width=1400,height=750) 
    # student_win.attributes('-fullscreen',True)
    student_win.iconbitmap("logo_icon.ico")
    
    frame=CTkFrame(student_win,width=1900,height=1000,fg_color="#66B3FF")
    frame.pack()
    #Home frame
    f0=CTkFrame(frame,width=1400,height=700,fg_color="#66B3FF")
    f0.place(x=140,y=30)
    #Dashboard
    f1=CTkFrame(frame,width=100,height=680,fg_color="white",border_width=3,corner_radius=15,border_color="black")
    f1.place(x=50,y=35)
    #logo
    photo=CTkImage(Image.open("logo.png"),size=(60,60))
    l1=CTkLabel(f1,image=photo,text=" ")
    l1.place(x=18,y=40)

    #home indicator
    home_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    home_indicate.place(x=7,y=150)
    
    search_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    search_indicate.place(x=7,y=250)
    
    schedule_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    schedule_indicate.place(x=7,y=350)

    complain_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    complain_indicate.place(x=7,y=450)
    
    attendance_indicate=CTkLabel(f1,fg_color="white",text=" ",height=55,width=2,corner_radius=9)
    attendance_indicate.place(x=7,y=550)
    
    #to initialize the student_win
    indicate(home_indicate,home_frame)
    #home button
    photo1=CTkImage(Image.open("home.png"),size=(50,50))
    b1=CTkButton(f1,command=lambda: indicate(home_indicate,home_frame),image=photo1,text="",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b1.place(x=17,y=150)
    #search button
    photo2=CTkImage(Image.open("search_logo.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(search_indicate,search_frame),image=photo2,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=250)
    ##taught_by button 
    photo3=CTkImage(Image.open("Schedule.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(schedule_indicate,academic_frame),image=photo3,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=350)
    #Complain button
    photo5=CTkImage(Image.open("report.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(complain_indicate,complain_frame),image=photo5,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=450)
    #Atttendance button
    photo5=CTkImage(Image.open("detailed_attend.png"),size=(50,50))
    b2=CTkButton(f1,command=lambda: indicate(attendance_indicate,attendance_frame),image=photo5,text=" ",hover_color="white",cursor="hand2",width=15,height=40,fg_color="white")
    b2.place(x=15,y=550)
    student_win.mainloop()

main_window()


