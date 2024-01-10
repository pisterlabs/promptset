# Importing
import datetime,wikipedia,webbrowser,os,random,requests,pyautogui,playsound, time,smtplib, ssl, csv, random,cv2,time ,keyboard
import bs4 as  sys,threading
import Annex,wolframalpha 
from ttkthemes import themed_tk 
import tkinter as tk
from tkinter import scrolledtext
from PIL import ImageTk,Image
import sqlite3,pyjokes,pywhatkit
from functools import partial
import calendar
from Main import TasksExecutor
from time import sleep 
import numpy as np 
from win32api import GetSystemMetrics 
from email.message import EmailMessage
import sounddevice as sd
import scipy.io.wavfile as wav 
try:
    app=wolframalpha.Client(" API KEY")  #API key for wolframalpha
except Exception as e:
    pass


#setting chrome path
chrome_path="C:\Program Files\Google\Chrome\Application\chrome.exe%s"

def there_exists(terms,query):
    for term in terms:
        if term in query:
            return True
 
def CommandsList():
    '''show the command to which voice assistant is registered with'''
    os.startfile('Commands List.txt') 

def clearScreen():
    ''' clear the scrollable text box'''
    SR.scrollable_text_clearing()  

def greet():
    conn = sqlite3.connect('Assistant.db')
    mycursor=conn.cursor()
    hour=int(datetime.datetime.now().hour)
    if hour>=4 and hour<12:
        mycursor.execute('select sentences from goodmorning')
        result=mycursor.fetchall()
        SR.speak(random.choice(result)[0])
    elif hour>=12 and hour<18:
        mycursor.execute('select sentences from goodafternoon')
        result=mycursor.fetchall()
        SR.speak(random.choice(result)[0])
    elif hour>=18 and hour<21:
        mycursor.execute('select sentences from goodevening')
        result=mycursor.fetchall()
        SR.speak(random.choice(result)[0])
    else:
        mycursor.execute('select sentences from night')
        result=mycursor.fetchall()
        SR.speak(random.choice(result)[0])
    conn.commit()
    conn.close()
    SR.speak("\nMyself Assistant. How may I help you?") 

import openai
from dotenv import load_dotenv

fileopen = open("config.txt","r")
API =fileopen.read()
fileopen.close()
openai.api_key = " "
load_dotenv()
completion = openai.Completion()        
 
def ReplyBrain(question,chat_log = None):
        FileLog = open("chat_log.txt","r")
        chat_log_template = FileLog.read()
        FileLog.close()
        if chat_log is None:
            chat_log = chat_log_template

        prompt = f'{chat_log}You : {question}\nAssistants: '
        response = completion.create(
            model = "text-davinci-002",
            prompt=prompt,
            temperature = 0.5,
            max_tokens = 60,
            top_p = 0.3,
            frequency_penalty = 0.5,
            presence_penalty = 0)
        answer = response.choices[0].text.strip()
        chat_log_template_update = chat_log_template + f"\nYou : {question} \nAssistant : {answer}"
        FileLog = open("chat_log.txt","w")
        FileLog.write(chat_log_template_update)
        FileLog.close()
        return answer

def mainframe():
    """Logic for execution task based on query"""
    SR.scrollable_text_clearing()
    greet()
    query_for_future=None
    try:
        while(True):
            query=SR.takeCommand().lower()          #converted the command in lower case of ease of matching

            #wikipedia search
            if there_exists(['search wikipedia','from wikipedia'],query):
                SR.speak("Searching wikipedia...")
                if 'search wikipedia' in query:
                    query=query.replace('search wikipedia','')
                    results=wikipedia.summary(query,sentences=2)
                    SR.speak("According to wikipedia:\n")
                    SR.speak(results)
                elif 'from wikipedia' in query:
                    query=query.replace('from wikipedia','')
                    results=wikipedia.summary(query,sentences=2)
                    SR.speak("According to wikipedia:\n")
                    SR.speak(results)
            elif there_exists(['wikipedia'],query):
                SR.speak("Searching wikipedia....")
                query=query.replace("wikipedia","")
                results=wikipedia.summary(query,sentences=2)
                SR.speak("According to wikipedia:\n")
                SR.speak(results)

            #jokes
            elif there_exists(['tell me joke','tell me a joke','tell me some jokes','i would like to hear some jokes',"i'd like to hear some jokes",
                            'can you please tell me some jokes','i want to hear a joke','i want to hear some jokes','please tell me some jokes',
                            'would like to hear some jokes','tell me more jokes'],query):
                SR.speak(pyjokes.get_joke(language="en", category="all"))
                query_for_future=query

            elif there_exists(['one more','one more please','tell me more','i would like to hear more of them','once more','once again','more','again'],query) and (query_for_future is not None):
                SR.speak(pyjokes.get_joke(language="en", category="all"))

            #calendar
            elif there_exists(['show me calendar','display calendar'],query):
                SR.updating_ST(calendar.calendar(2023))

            #google, youtube and location
            #playing on youtube
            elif there_exists(["play"],query):
                query=str(query).replace("Play","")#.replace("search on youtube","")
                pywhatkit.playonyt((query))
                SR.speak("Your video Has Been Started! , Enjoy Sir!")
                return True
            
            elif there_exists(["google"],query):
                query = query.replace("google", "").strip()  # Remove "google" and any leading/trailing spaces
                pywhatkit.search(query)
                SR.speak("This site I have found.")
                return True 
             
            elif there_exists(['find location of','show location of','find location for','show location for'],query):
                if 'of' in query:
                    url='https://google.nl/maps/place/'+query[query.find('of')+3:]+'/&amp'
                    webbrowser.get(chrome_path).open(url)
                    break
                elif 'for' in query:
                    url='https://google.nl/maps/place/'+query[query.find('for')+4:]+'/&amp'
                    webbrowser.get(chrome_path).open(url)
                    break
            elif there_exists(["what is my exact location","What is my location","my current location","exact current location"],query):
                url = "https://www.google.com/maps/search/Where+am+I+?/"
                webbrowser.get().open(url)
                SR.speak("Showing your current location on google maps...")
                break
            

            #who is 
            #elif 'who is' in query or'who the heck is' in query or 'who the hell is' in query or 'who is this' in query or "what is" in query or "how" in query or "question" in query or "i have query" in query or "what" in query:
            elif there_exists(["question","I have a quey"], query):  
                SR.speak("Tell me Query")
                data=SR.takeCommand()
                Reply = ReplyBrain(data)
                SR.speak(Reply)
                

            # top 5 news
            elif there_exists(['top 5 news','top five news','listen some news','news of today'],query):
                news=Annex.News(scrollable_text)
                news.show() 

            #what is meant by
            elif there_exists(['what is meant by','what is mean by'],query):
                results=wikipedia.summary(query,sentences=2)
                SR.speak("According to wikipedia:\n")
                SR.speak(results)

            #taking photo
            elif there_exists(['take a photo','take a selfie','take my photo','take photo','take selfie','one photo please','click a photo'],query):
                takephoto=Annex.camera()
                Location=takephoto.takePhoto()
                os.startfile(Location)
                del takephoto
                SR.speak("Captured picture is stored in Camera folder.")

            #makig note
            elif there_exists(['make a note','take note','take a note','note it down','make note','remember this as note','open notepad and write'],query):
                SR.speak("What would you like to write down?")
                data=SR.takeCommand()
                n=Annex.note()
                n.Note(data)
                SR.speak("I have a made a note of that.")
                break

            elif there_exists(['notes'],query):
                
                def create_note(note_title, note_content):
                    try:
        
                        directory_path = "C:\\Users\\shubh\\OneDrive\\Pictures\\Notes"  # Get the directory path from the user
                        SR.speak("Tell Me The File Name")
                        file_name =SR.takeCommand()  # Get the file name from the user
                        file_path = os.path.join(directory_path, f"{file_name}.txt")

                        with open(file_path, "a") as f:
                            f.write(f"### {note_title} ###\n")
                            f.write(note_content + "\n")
                            f.write("=" * 30 + "\n")
                        SR.speak("Note saved.")
                        os.startfile(directory_path)
                        SR.speak("Here Is Your Note.")
                    except Exception as e:
                        SR.speak(f"An error occurred: {str(e)}")

                SR.speak("Tell Me the Note Title")
                note_title = SR.takeCommand()
                SR.speak("Tell Me The Note Content")
                note_content = SR.takeCommand()
                create_note(note_title, note_content)
                continue

            #flipping coin
            elif there_exists(["toss a coin","flip a coin","toss"],query):
                moves=["head", "tails"]
                cmove=random.choice(moves)
                playsound.playsound('quarter spin flac.mp3')
                SR.speak("It's " + cmove)

            #time and date
            elif there_exists(['the time'],query):
                strTime =datetime.datetime.now().strftime("%H:%M:%S")
                SR.speak(f"Sir, the time is {strTime}")
            elif there_exists(['the date'],query):
                strDay=datetime.date.today().strftime("%B %d, %Y")
                SR.speak(f"Today is {strDay}")
           

            #opening software applications
            elif there_exists(["open"],query):
                Nameoftheapp = query.replace("open ","")
                pyautogui.press('win')
                sleep(1)
                keyboard.write(Nameoftheapp)
                sleep(1)
                keyboard.press('enter')
                sleep(0.5) 
                SR.speak("Command is completed.") 
                return True
            
            elif there_exists(['close','Close'],query):
                SR.speak("Ok , Wait A second!")
                SR.speak("Tell Me Which Application You Want To close")
                Query = SR.takeCommand() 
                if "ms word" or 'Ms Word' in Query:
                    os.system("TASKKILL /F /im WINWORD.EXE")
                elif 'Chrome' or 'chrome'in Query:
                    os.system("TASKKILL /f /im chrome.exe")
                elif 'Power Point' or 'power point' or 'power point' in Query:
                    os.system("TASKKILL /F /im POWERPNT.EXE")
                elif 'code' or 'code' in Query:
                    os.system("TASKKILL /F /im Code.exe")
                elif 'MS Browser' or 'ms browser' in Query:
                    os.system("TASKKILL /F /im msedge.exe")
                elif 'Excel' or 'excel' in Query:
                    os.system("TASKKILL /F /im EXCEL.EXE")  
                elif 'Brave' or 'brave' in Query:
                    os.system("TASKKILL /F /im brave.exe")
                SR.speak("Your Command Has Been Succesfully Completed!") 
        
            elif there_exists(['show me setting window' ],query):
                SR.speak("Opening my Setting window..")
                sett_wind=Annex.SettingWindow()
                sett_wind.settingWindow(root)
                break

            #password generator
            elif there_exists(['suggest me a password','password suggestion','i want a password'],query):
                m3=Annex.PasswordGenerator()
                m3.givePSWD(scrollable_text)
                del m3  
            
            #screeshot
            elif there_exists(['take screenshot','take a screenshot','screenshot please','capture my screen'],query):
                SR.speak("Ok , What Should I Name That File ?")
                path = SR.takeCommand()
                path1name = path + ".png"
                path1 = "C:\\Users\\shubh\\OneDrive\\Pictures\\Screenshots\\"+ path1name
                kk = pyautogui.screenshot()
                kk.save(path1)
                os.startfile("C:\\Users\\shubh\\OneDrive\\Pictures\\Screenshots")
                SR.speak("Here Is Your ScreenShot")
                continue

            #voice recorder
            elif there_exists( ['voice recorder'] , query):
                SR.speak("Ok, Sir, tell me the time duration")
                Time = float(SR.takeCommand())  # Convert to float
                # Set the parameters for audio recording
                duration = Time  # Recording duration in seconds
                sample_rate = 44100  # Sample rate (samples per second)
                SR.speak("Sir, Tell me the File Name")
                File_Name = SR.takeCommand()
                output_file = f"C:\\Users\\shubh\\OneDrive\\Pictures\\Audio\\{File_Name}.wav"

                try:
                    # Start recording
                    SR.speak(f"Recording audio for {duration} seconds...")
                    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
                    sd.wait()

                    # Save the recorded audio to a WAV file
                    wav.write(output_file, sample_rate, audio_data)

                    SR.speak(f"Audio recording saved")
        
                    # Speak "Here is your audio" after successful recording
                    SR.speak("Audio Recording finished. Here is your audio.")
        
                    # Open the folder containing the recorded audio
                    os.startfile("C:\\Users\\shubh\\OneDrive\\Pictures\\Audio")
                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    SR.speak("An error occurred during audio recording.")
                    continue

            
            
            

            #weather report
            elif there_exists(['weather report'],query):
                    # Enter your API key from the Open Weather website
                    APIKey = 'Weather API KEY'

                    # store the base url
                    BaseURL = 'https://api.openweathermap.org/data/2.5/weather'

                    # view the specific data, rather than hard to read dictionary and list data
                    # Store and print greeting
                    SR.speak('Welcome to your weather tracker! Which City do you want to view?')
                    # Input the relevant City name
                    SR.speak('Tell me a City name')
                    City=SR.takeCommand()
                    RequestURL = f'{BaseURL}?appid={APIKey}&q={City}'
                    Response = requests.get (RequestURL)
                    # If the response is 'ok' then get the whole JSON weather data as a Python dictionary
                    if Response.status_code == 200:
                        Data = Response.json ()
                        weather = Data['weather'][0]['description']
                        temperature = round(Data['main']['temp']-273.15,2)
                        sunrise = datetime.datetime.utcfromtimestamp(Data['sys']['sunrise']+ Data['timezone'])
                        sunset = datetime.datetime.utcfromtimestamp(Data['sys']['sunset']+ Data['timezone'])
                        SR.speak(f'weather summery:{weather}')
                        SR.speak(f'temperature is {temperature} celsius')
                        SR.speak(f'sunrise time is {sunrise}')
                        SR.speak(f'sunset time is s{sunset}')

                    else:
                        print("hmm...that doesn't look quite right, try again")

            elif there_exists(['screen recording'],query):
                # Get screen width and height
                width = GetSystemMetrics(0)
                height = GetSystemMetrics(1)
                dim = (width, height)
    
                # Define the codec and create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
                # Prompt the user for the recording duration
                SR.speak("How many seconds do you want to record?")
                Time = float(SR.takeCommand()) 
                duration = Time  # Convert the user input to a float
    
                # Prompt the user for the file name
                SR.speak("Tell Me File Name")
                FileName = SR.takeCommand()
    
                # Create the output VideoWriter with the specified file name
                output = cv2.VideoWriter(f"C:\\Users\\shubh\\OneDrive\\Pictures\\Recording\\{FileName}.avi", fourcc, 20.0, dim)
    
                end_time = time.time() + duration
    
                try:
                    while True:
                        # Capture the screen frame
                        image = pyautogui.screenshot()
                        frame = np.array(image)
            
                        # Convert the color format to BGR (compatible with OpenCV)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
                        # Write the frame to the video file
                        output.write(frame)
            
                        # Check if the recording duration has elapsed
                        if time.time() > end_time:
                            break
                except Exception as e:
                    SR.speak(f"An error occurred: {str(e)}")
    
                # Release the VideoWriter
                output.release()
                SR.speak("Recording finished.")
                os.startfile("C:\\Users\\shubh\\OneDrive\\Pictures\\Recording")
                SR.speak("Here Is Your Screen Recording")

            #shutting down system
            elif there_exists(['exit','quit','shutdown', 'shut down'],query):
                SR.speak("shutting down")
                sys.exit()

            elif there_exists(['none'],query):
                pass 
 

            #what is the capital
            elif there_exists(['what is the capital of','capital of','capital city of'],query):
                try:
                    res=app.query(query)
                    SR.speak(next(res.results).text)
                except:
                    print("Sorry, but there is a little problem while fetching the result.")

            elif there_exists(['temperature'],query):
                try:
                    res=app.query(query)
                    SR.speak(next(res.results).text)
                except:
                    print("Internet Connection Error")

            elif there_exists(['+','-','*','x','/','plus','add','minus','subtract','divide','multiply','divided','multiplied'],query):
                try:
                    res=app.query(query)
                    SR.speak(next(res.results).text)
                except:
                    print("Internet Connection Error")

            elif there_exists( ["mail"] , query):
                SR.speak("Please,Tell Me Your Reply To Email Id")
                replyto= SR.takeCommand()
                SR.speak("Subject Of Your Email")
                subject = SR.takeCommand()
                SR.speak("PLease Tell Me Email Name")
                name = SR.takeCommand()
                counter = {}
                with open("user.csv") as f:
                    data = [row for row in csv.reader(f)]

                # file_list = ['emails/message' + str(i) + '.txt' for i in range(1,11)] # Multipale Files
                file_list = ['emails\\message1.txt'] #Singale Email

                with open('mails.csv', 'r') as csvfile:
                    datareader = csv.reader(csvfile)
                    for row in datareader:
                        random_user = random.choice(data)
                        sender = random_user[0]
                        password = random_user[1]
        
                        if sender not in counter:
                            counter[sender] = 0
        
                        if counter[sender] >= 500:
                            continue
            
                        try:
                            context = ssl.create_default_context()
                            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
                            server.login(sender, password)
                            em =EmailMessage()
                            em['from'] = f'{name} <{sender}>'
                            em['Reply-To'] = replyto
                            em['To'] = row
                            em['subject'] = subject
                            random_file = random.choice(file_list)
                            with open(random_file, 'r') as file:
                                html_msg = file.read()
                            em.add_alternative(html_msg, subtype='html')
                            server.send_message(em)
                            counter[sender] += 1
                            print(counter[sender], " emails sent", "From ", sender,  "To ", row ,"File ", random_file)
                            with open("mails.csv", "r") as file:
                                reader = csv.reader(file)
                                rows = list(reader)
                                rows = rows[1:]
                            if rows:
                                with open("mails.csv", "w", newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerows(rows)
                            server.close()
                        except Exception as e:
                            SR.speak(f"Error sending email From {sender} to {row}:", e )
                            with open("mails.csv", "r") as file:
                                reader = csv.reader(file)
                                rows = list(reader)
                                rows = rows[1:]
                            if rows:
                                with open("mails.csv", "w", newline='') as file:
                                    writer = csv.writer(file)
                                    writer.writerows(rows)

                SR.speak("Emails Sent")
                for sender, count in counter.items():
                    SR.speak(f"{sender}: {count}")

            
            elif there_exists( ["Gmail","gmail","i want to send gmail","I want to send Gmail"],query):
                    SR.speak("Tell Me Subject ")
                    sleep(3)
                    subject = SR.takeCommand()
                    SR.speak("Tell Me body ")
                    sleep(3)
                    body = SR.takeCommand()
                    SR.speak("Tell Me Sender_email ")
                    sleep(3)
                    sender_email ="Shubhamthorat2001@gmail.com" #SR.takeCommand()
                    SR.speak("Tell Me receiver_email ")
                    sleep(3)
                    receiver_email = "Shubhamthorat070@gmail.com" #SR.takeCommand()
                    SR.speak("Tell Me password")
                    sleep(3)
                    password = "lbzwdpvtsyozjjbq"

                    SR.speak("I am working....")
                    sleep(3)

                    message = EmailMessage()
                    message["From"] = sender_email
                    message['To']= receiver_email
                    message['Subject']= subject
                    message.set_content(body)


                    context = ssl.create_default_context()
                    SR.speak("sending Email!")
                    with smtplib.SMTP_SSL("smtp.gmail.com",465, context= context) as server:
                        server.login(sender_email,password)
                        server.sendmail(sender_email,receiver_email,message.as_string())

                    SR.speak("sucessfull")

            else:
                ReturnData=TasksExecutor(query)
                SR.speak(ReturnData)
    except Exception as e:
        pass

def gen(n):
    for i in range(n):
        yield i

class MainframeThread(threading.Thread):
    def __init__(self, threadID, name):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
    def run(self):
        mainframe()

def Launching_thread():
    Thread_ID=gen(1000)
    global MainframeThread_object
    MainframeThread_object=MainframeThread(Thread_ID.__next__(),"Mainframe")
    MainframeThread_object.start()

if __name__=="__main__":
        #tkinter code
        root=themed_tk.ThemedTk()
        root.set_theme("winnative")
        root.geometry("{}x{}+{}+{}".format(745,360,int(root.winfo_screenwidth()/2 - 745/2),int(root.winfo_screenheight()/2 - 360/2)))
        root.resizable(0,0)
        root.title("Assistant")
        root.iconbitmap('Assistant.ico')
        root.configure(bg='#2c4557')
        scrollable_text=scrolledtext.ScrolledText(root,state='disabled',height=15,width=87,relief='sunken',bd=5,wrap=tk.WORD,bg='#add8e6',fg='#800000')
        scrollable_text.place(x=10,y=10)
        mic_img=Image.open("Mic.png") 
        mic_img=mic_img.resize((55,55))
        mic_img=ImageTk.PhotoImage(mic_img)
        Speak_label=tk.Label(root,text="SPEAK:",fg="#FFD700",font='"Times New Roman" 12 ',borderwidth=0,bg='#2c4557')
        Speak_label.place(x=250,y=300)
        """Setting up objects"""
        SR=Annex.SpeakRecog(scrollable_text)    #Speak and Recognition class instance
        Listen_Button=tk.Button(root,image=mic_img,borderwidth=0,activebackground='#2c4557',bg='#2c4557',command=Launching_thread)
        Listen_Button.place(x=330,y=280)
        myMenu=tk.Menu(root)
        m1=tk.Menu(myMenu,tearoff=0) #tearoff=0 means the submenu can't be teared of from the window
        m1.add_command(label='Commands List',command=CommandsList)
        myMenu.add_cascade(label="Help",menu=m1)
        stng_win=Annex.SettingWindow()
        myMenu.add_cascade(label="Settings",command=partial(stng_win.settingWindow,root))
        myMenu.add_cascade(label="Clear Screen",command=clearScreen)
        root.config(menu=myMenu)
        root.mainloop()