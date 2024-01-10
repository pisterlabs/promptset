import tkinter as tk
import pyttsx3
import speech_recognition as sr
from tkinter import messagebox
import subprocess
import boto3
import tkinter.filedialog as filedialog
from cvzone.HandTrackingModule import HandDetector
import requests
from PIL import Image, ImageTk
import os
import re
import cv2
import time
import qrcode
import webbrowser
import tkinter.simpledialog as simpledialog
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import numpy as np
from math import hypot
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import requests
from tkinter import ttk
import pywhatkit as py






def do_something():
    messagebox.showinfo("Info", "Team 9 - Stambh")

def system_app():
    system_window = tk.Toplevel(root)
    system_window.title("System Apps")
    system_window.geometry("400x275")
    system_window.configure(bg="lightgray")
    
    system_label = tk.Label(system_window, text="System Apps", font=("Arial", 20,))
    system_label.pack(pady=10)

    notepad_button = tk.Button(system_window, text="Open Notepad", width=25, fg='Black', command=open_notepad)
    notepad_button.pack(pady=10)

    calculator_button = tk.Button(system_window, text="Open Calculator", width=25, fg='Black', command=open_calculator)
    calculator_button.pack(pady=10)
    
    explorer_button= tk.Button(system_window, text="Explorer", width=25, command=open_explorer)
    explorer_button.pack(pady=10)

    close_button = tk.Button(system_window, text="Close",width=25, command=system_window.destroy)
    close_button.pack(pady=10)
    
def aws_operations_window():
    aws_window = tk.Toplevel(root)
    aws_window.title("AWS Operations")
    aws_window.geometry("400x350")
    aws_window.configure(bg="lightgray")
    
    aws_label = tk.Label(aws_window, text="AWS Operations", font=("Arial", 20, "bold"))
    aws_label.pack(pady=10)

    ec2_button = tk.Button(aws_window, text="Create EC2 Instance",width=25, command=open_ec2_instance)
    ec2_button.pack(pady=15)

    s3_button = tk.Button(aws_window, text="Create S3 Bucket",width=25, command=create_s3_bucket)
    s3_button.pack(pady=15)
    
    s3_button = tk.Button(aws_window, text="Upload to S3",width=25, command=upload_to_s3)
    s3_button.pack(pady=15)
    
    list_ec2_button = tk.Button(aws_window, text="List EC2 Instances", width=25, command=list_ec2_instances)
    list_ec2_button.pack(pady=15)

    close_button = tk.Button(aws_window, text="Close",width=25, command=aws_window.destroy)
    close_button.pack(pady=10)

def on_exit():
    if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
        root.destroy()

def open_notepad():
    subprocess.Popen("notepad.exe")

def open_calculator():
    subprocess.Popen("calc.exe")
    
def open_explorer():
    subprocess.Popen("explorer.exe")

import boto3

def list_ec2_instances():
    try:
        ec2 = boto3.client("ec2")
        response = ec2.describe_instances()
        instances = response["Reservations"]

        if not instances:
            messagebox.showinfo("No EC2 Instances", "No EC2 instances found.")
        else:
            instance_info = "\n".join([f"ID: {instance['Instances'][0]['InstanceId']}, "
                                       f"State: {instance['Instances'][0]['State']['Name']}"
                                       for instance in instances])
            messagebox.showinfo("EC2 Instances", f"List of EC2 instances:\n{instance_info}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to list EC2 instances: {e}")

def open_ec2_instance():
    response = messagebox.askyesno("AWS EC2 Instance", "Do you want to create an EC2 instance?")
    if response:
        myec2 = boto3.client("ec2")
        response = myec2.run_instances(  
            ImageId='ami-0ded8326293d3201b', 
            InstanceType='t2.micro',
            MaxCount=1,
            MinCount=1
        )
        print(response)

def create_s3_bucket():
    response = messagebox.askyesno("AWS S3 Bucket", "Do you want to create an S3 bucket?")
    if response:
        s3 = boto3.client('s3')
        s3 = s3.create_bucket(
            Bucket='arpit230709845',
            ACL='private',
            CreateBucketConfiguration={
                'LocationConstraint': 'ap-south-1'
            }
        )
        print("Bucket created successfully with the following response:")
        print(s3)
        print("Bucket 'new' was created in the 'ap-south-1' region.")
        
def upload_to_s3():
    bucket_name = simpledialog.askstring("Upload to S3 Bucket", "Enter the bucket name:")
    if bucket_name:
        file_path = filedialog.askopenfilename(title="Select a file to upload")
        if file_path:
            try:
                s3 = boto3.client("s3")
                file_name = os.path.basename(file_path)
                s3.upload_file(file_path, bucket_name, file_name)
                messagebox.showinfo("Upload Successful", f"File '{file_name}' uploaded to '{bucket_name}'")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to upload file: {e}")

def open_youtube():
    song_name = simpledialog.askstring("Open YouTube", "Enter the name of your favorite song:")
    if song_name:
        search_query = song_name.replace("", "+")
        webbrowser.open(f"https://www.youtube.com/results?search_query={search_query}")


def google_search():
    search_query = simpledialog.askstring("Google Search", "Enter your search query:")
    if search_query:
        webbrowser.open(f"https://www.google.com/search?q={search_query}")
        

def deaf_people_help():
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  
    engine.setProperty("volume", 1.0)
    cap=cv2.VideoCapture(0)
    model=HandDetector()
    while True:
        status,photo=cap.read()
        cv2.imshow("pic1",photo)
        if cv2.waitKey(10)==13:
            break

        hand=model.findHands(photo,draw=False)
        if hand:
            handPhoto=hand[0]
#    print(handPhoto)
            fingerlist=model.fingersUp(handPhoto)        
            if fingerlist==[0,1,1,1,1]:
                engine.say("namaste")
                engine.runAndWait()
                print("namaste")
                time.sleep(2)
            elif fingerlist==[1,0,0,0,0]:
                engine.say("Good job")
                engine.runAndWait()
                print("Good job")
                time.sleep(2)
            elif fingerlist==[0,1,1,0,0]:
                engine.say("Pleasure meeting with you")
                engine.runAndWait()
                print("Pleasure meeting with you")
                time.sleep(2)
            elif fingerlist==[0,1,1,1,0]:
                engine.say("Perfect")
                engine.runAndWait()
                print("Perfect")
                time.sleep(2)
            elif fingerlist==[1,1,0,0,1]:
                engine.say("I love Vimal Sir")
                engine.runAndWait()
                print("I love Vimal Sir")
                time.sleep(2)
            elif fingerlist==[0,0,0,0,0]:
                engine.say("Sorry")
                engine.runAndWait()
                print("Sorry")
                time.sleep(2)
            elif fingerlist==[0,1,0,0,0]:
                engine.say("Help")
                engine.runAndWait()
                print("Help")
                time.sleep(2)
            else:
                print("dont support")
                time.sleep(2)
        


  
    cv2.destroyAllWindows()
    cap.release()
        

def capture_video():
    if messagebox.askyesno("Exit", "Want to Capture ?"):
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("captured_video.avi", fourcc, 20.0, (640, 480))

    while True:
        re, frame = cap.read()
        if not ret:
            break
        out.write(frame)

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    messagebox.showinfo("Video Captured", "Video captured and saved as 'captured_video.avi'")
    
import requests

def get_weather():
    city = simpledialog.askstring("Weather Update", "Enter city name:")
    if city:
        api_key = "b468e8a2145b49204e0af7d5594d376a"
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        weather_data = response.json()
        if weather_data["cod"] == 200:
            weather_info = f"Weather in {city}: {weather_data['weather'][0]['description']}\n" \
                           f"Temperature: {weather_data['main']['temp']}°C\n" \
                           f"Humidity: {weather_data['main']['humidity']}%"
            messagebox.showinfo("Weather Update", weather_info)
        else:
            messagebox.showerror("Error", "Failed to fetch weather data.")

def Image_swaping():
    import cv2

# Load the images
    image = cv2.imread("tuntun.jpeg")
    image2 = cv2.imread("kalia.jpeg")

# Perform the face swap
    image[40:200, 130:300] = image2[70:230, 130:300]

# Display the image with swapped face
    cv2.imshow("Face Swapped Image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()
        


def Volume_by_hand_gesture():
    cap = cv2.VideoCapture(0)  # Checks for camera

    mpHands = mp.solutions.hands  # detects hand/finger
    hands = mpHands.Hands()  # complete the initialization configuration of hands
    mpDraw = mp.solutions.drawing_utils

    # To access the speaker through the library pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volbar = 400
    volper = 0

    volMin, volMax = volume.GetVolumeRange()[:2]

    while True:
        success, img = cap.read()  # If the camera works, capture an image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to rgb

        # Collection of gesture information
        results = hands.process(imgRGB)  # completes the image processing.

        lmList = []  # empty list
        if results.multi_hand_landmarks:  # list of all hands detected.
            # By accessing the list, we can get the information of each hand's corresponding flag bit
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):  # adding counter and returning it
                    # Get finger joint points
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])  # adding to the empty list 'lmList'
                mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

        if lmList != []:
            # getting the value at a point
            # x      # y
            x1, y1 = lmList[4][1], lmList[4][2]  # thumb
            x2, y2 = lmList[8][1], lmList[8][2]  # index finger
            # creating a circle at the tips of thumb and index finger
            cv2.circle(img, (x1, y1), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
            cv2.circle(img, (x2, y2), 13, (255, 0, 0), cv2.FILLED)  # image #fingers #radius #rgb
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # create a line b/w tips of index finger and thumb

            length = hypot(x2 - x1, y2 - y1)  # distance b/w tips using hypotenuse
            # from numpy we find our length,by converting hand range in terms of volume range ie b/w -63.5 to 0
            vol = np.interp(length, [30, 350], [volMin, volMax])
            volbar = np.interp(length, [30, 350], [400, 150])
            volper = np.interp(length, [30, 350], [0, 100])

            print(vol, int(length))
            volume.SetMasterVolumeLevel(vol, None)

            # Hand range 30 - 350
            # Volume range -63.5 - 0.0
            # creating a volume bar for volume level
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 4)  # vid ,initial position ,ending position ,rgb ,thickness
            cv2.rectangle(img, (50, int(volbar)), (85, 400), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, f"{int(volper)}%", (10, 40), cv2.FONT_ITALIC, 1, (0, 255, 98), 3)
            # tell the volume percentage ,location,font of text,length,rgb color,thickness
        cv2.imshow('Image', img)  # Show the video
        if cv2.waitKey(1) == 13:  # By using spacebar delay will stop
            break

    cap.release()  # stop cam
    cv2.destroyAllWindows()



        


def generate_qr_code():
    data = simpledialog.askstring("QR Code Generator", "Enter the text or URL to encode:")
    if data:
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img.save("qr_code.png")
        img.show()
        
def langchain_SERP_Search():
    print("Google search with open ai")
    from langchain.llms import OpenAI
    myllm=OpenAI(
        model='text-davinci-003',
        temperature=1,
        openai_api_key="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"
    )

    myserpkey="4a2bb0b7f2ceb8972030d52c446c5a74857cb25bb1b085555a12c5d37c446047"

    import os
    os.environ['SERPAPI_API_KEY']=myserpkey

    from langchain.agents import load_tools
    myserptool=load_tools(tool_names=['serpapi'])

    from langchain.agents import AgentType
    from langchain.agents import initialize_agent
    mygooglechain=initialize_agent(
        llm=myllm,
        tools=myserptool,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    mygooglechain.run("when chandrayan 3 launched")
    
def langchain_Brave_Search():
    print("Brave Search starts")
    from langchain.llms import OpenAI
    from langchain.tools import BraveSearch

    myllm = OpenAI(
        model='text-davinci-003',
        temperature=1,
        openai_api_key="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"
    )
    
    api_key = "BSAv1neIuQOsxqOyy0sEe_ie2zD_n_V"
    tool = BraveSearch.from_api_key(api_key=api_key, search_kwargs={"count": 5})
    
    from langchain.agents import AgentType
    from langchain.agents import initialize_agent

    mygooglechain = initialize_agent(
        llm=myllm,
        tools=[tool],
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    mygooglechain.run("when chandrayan 3 launched")    
def playing_with_pixels():
    image3=cv2.imread("blank.jpg.jpeg")
    photo1_resized = cv2.resize(image3, (700, 500))
    print(photo1_resized.shape)
#start
    photo1_resized[0:1,:,] = [0]
#road
    photo1_resized[400:500,:,] = [0]
#road white strip
    for x in range(0,700,100):photo1_resized[440:460,x:x+50,]=[255]
#poll
    photo1_resized[250:400,20:22]=[0]
    photo1_resized[250:252,3:40]=[0]
    photo1_resized[247:253,0:6]=[0,0,255]
    photo1_resized[247:253,37:43]=[0,0,255]
#building structure
    photo1_resized[100:400,50:250]=[0,255,255]#BRG
#building window
    for x in range(120,400,60):photo1_resized[x:x+30,65:130]=[36,x+28,237]
    for x in range(120,400,60):photo1_resized[x:x+30,155:230]=[36,x+28,237]

#person
    photo1_resized[390:400,270:271]=[0]
    photo1_resized[390:400,274:275]=[0]

    photo1_resized[389:390,266:278]=[0]

    photo1_resized[380:390,265:266]=[0]
    photo1_resized[380:390,277:278]=[0]

    photo1_resized[385:386,260:265]=[0]
    photo1_resized[385:386,277:282]=[0]

    photo1_resized[379:380,266:278]=[0]

    photo1_resized[373:379,270:271]=[0]
    photo1_resized[373:379,274:275]=[0]

    photo1_resized[372:373,270:274]=[0]

    photo1_resized[375:378,273:274]=[36,28,237]

#poll
    photo1_resized[250:400,300:302]=[0]
    photo1_resized[250:252,280:320]=[0]
    photo1_resized[247:253,277:283]=[0,0,255]
    photo1_resized[247:253,317:323]=[0,0,255]

#glass building
    photo1_resized[150:400,350:600]=[234,217,153]#BRG
    for x in range(370,590,30):
        photo1_resized[150:400,x:x+1]=[0]
    for x in range(180,400,30):
        photo1_resized[x:x+1,350:600]=[0]
#poll
    photo1_resized[250:400,650:652]=[0]
    photo1_resized[250:252,630:670]=[0]
    photo1_resized[247:253,627:633]=[0,0,255]
    photo1_resized[247:253,667:673]=[0,0,255]


    cv2.imshow("divu",photo1_resized)
    cv2.waitKey()
    cv2.destroyAllWindows()
    

    
def object_detection():
   

    def detect_labels(image_path):
        aws_access_key = 'AKIATXKPLAMCGIVSCIPN'
        aws_secret_key = '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'

        client = boto3.client('rekognition', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name='ap-south-1')

    
        with open(image_path, 'rb') as image_file:
            image_bytes = image_file.read()

        response = client.detect_labels(Image={'Bytes': image_bytes})

        return response['Labels']

    def main():
        image_path = 'Messireal.jpg'
        labels = detect_labels(image_path)
    
        print("Labels in the image:")
        for label in labels:
            print(f"- {label['Name']} (Confidence: {label['Confidence']:.2f}%)")

    if __name__ == "__main__":
        main()

    
def Tik_Tac_TOE():
    
    def sum(a, b, c ):
        return a + b + c

    def printBoard(xState, zState):
        zero = 'X' if xState[0] else ('O' if zState[0] else 0)
        one = 'X' if xState[1] else ('O' if zState[1] else 1)
        two = 'X' if xState[2] else ('O' if zState[2] else 2)
        three = 'X' if xState[3] else ('O' if zState[3] else 3)
        four = 'X' if xState[4] else ('O' if zState[4] else 4)
        five = 'X' if xState[5] else ('O' if zState[5] else 5)
        six = 'X' if xState[6] else ('O' if zState[6] else 6)
        seven = 'X' if xState[7] else ('O' if zState[7] else 7)
        eight = 'X' if xState[8] else ('O' if zState[8] else 8)
        print(f"{zero} | {one} | {two} ")
        print(f"--|---|---")
        print(f"{three} | {four} | {five} ")
        print(f"--|---|---")
        print(f"{six} | {seven} | {eight} ") 

    def checkWin(xState, zState):
        wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        for win in wins:
            if(sum(xState[win[0]], xState[win[1]], xState[win[2]]) == 3):
                print("X Won the match")
                return 1
            if(sum(zState[win[0]], zState[win[1]], zState[win[2]]) == 3):
                print("O Won the match")
                return 0
        return -1
    
    if __name__ == "__main__":
        xState = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        zState = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        turn = 1 # 1 for X and 0 for O
        print("Welcome to Tic Tac Toe")
        while(True):
            printBoard(xState, zState)
            if(turn == 1):
                print("X's Chance")
                value = int(input("Please enter a value: "))
                xState[value] = 1
            else:
                print("O's Chance")
                value = int(input("Please enter a value: "))
                zState[value] = 1
            cwin = checkWin(xState, zState)
            if(cwin != -1):
                print("Match over")
                break
        
            turn = 1 - turn

def geo_coordinates():

    # get gps coordinates from geopy
    import json

# import urlopen from urllib.request
    from urllib.request import urlopen

# url to get ipaddress
    urlopen("http://ipinfo.io/json")

# load data into array
    data = json.load(urlopen("http://ipinfo.io/json"))

# lattitude
    lat = data['loc'].split(',')[0]

# longitude
    lon = data['loc'].split(',')[1]

    print(lat, lon)
    

def whatsapp_message():
    

    py.sendwhatmsg_instantly("+918813040675", "Hello, this is an instant WhatsApp message sent using pywhatkit!" )
    

        
# Create the main Tkinter window
root = tk.Tk()
root.title("Menu")
root.geometry("1200x750")
root.configure(bg="purple")
title_font = ("Arial", 24, "italic")
root.iconbitmap(r"E:\Summer_Pregram_2023_Offline\Python\practice\menu.png")

# Welcome Text
title_label = tk.Label(root, text="Python Menu", font=title_font, fg="black")
title_label.grid(row=0 , column=0 ,padx=20 , pady=40)

# Welcome Text
#title_label = tk.Label(root, text="Select Options Below", font=("Arial", 18, "italic"), fg="black")
#title_label.pack(pady=10)

notepad_button = tk.Button(root, text="System Apps", width=25, fg='Black', command=system_app)
notepad_button.grid(row=1 , column=0 ,padx=20 , pady=40)

capture_video_button = tk.Button(root, text="Capture Video", width=25, command=capture_video)
capture_video_button.grid(row=1 , column=1 ,padx=20 , pady=20)

weather_button = tk.Button(root, text="Weather Update", width=25, command=get_weather)
weather_button.grid(row=1 , column=2 ,padx=30 , pady=20)

aws_button = tk.Button(root, text="AWS Operations", width=25, command=aws_operations_window)
aws_button.grid(row=1 , column=3 ,padx=10 , pady=20)

browser_button = tk.Button(root, text="Youtube", width=25, command=open_youtube)
browser_button.grid(row=2 , column=0 ,padx=20 , pady=20)

google_button = tk.Button(root, text="Google Search", width=25, command=google_search)
google_button.grid(row=2 , column=1 ,padx=30 , pady=20)

deaf_people_help = tk.Button(root, text="Communicate With Deaf People", width=25, command=deaf_people_help)
deaf_people_help.grid(row=2 , column=2 ,padx=20 , pady=20)

Image_swaping_button = tk.Button(root, text="Image Swaping", width=25, command=Image_swaping)
Image_swaping_button.grid(row=2 , column=3 ,padx=10 , pady=20)


Volume_by_hand_gesture_button = tk.Button(root, text="Volume by hand gesture", width=25, command=Volume_by_hand_gesture)
Volume_by_hand_gesture_button.grid(row=3 , column=0 ,padx=20 , pady=20)


qr_code_button = tk.Button(root, text="Generate QR Code", width=25, command=generate_qr_code)
qr_code_button.grid(row=3 , column=1 ,padx=30 , pady=20)

langchain_SERP_Search_button = tk.Button(root, text="Langchain SERP API Search", width=25, command=langchain_SERP_Search)
langchain_SERP_Search_button.grid(row=3 , column=2 ,padx=20 , pady=20)

langchain_Brave_Search_button = tk.Button(root, text="Langchain Brave Search", width=25, command=langchain_Brave_Search)
langchain_Brave_Search_button.grid(row=3 , column=3 ,padx=30 , pady=20)


playing_with_pixels_button = tk.Button(root, text="Playing with pixels", width=25, command=playing_with_pixels)
playing_with_pixels_button.grid(row=4 , column=0 ,padx=20 , pady=20)

object_detection_button= tk.Button(root, text="Object Detection", width=25, command=object_detection)
object_detection_button.grid(row=4 , column=1 ,padx=30 , pady=20)

Tik_Tac_TOE_button= tk.Button(root, text="Tik Tac Toe", width=25, command=Tik_Tac_TOE)
Tik_Tac_TOE_button.grid(row=4 , column=2 ,padx=20 , pady=20)


geo_coordinates_button= tk.Button(root, text="Geo Coordinates", width=25, command=geo_coordinates)
geo_coordinates_button.grid(row=4 , column=3 ,padx=20 , pady=20)

whatsapp_button= tk.Button(root, text="Whatsapp Message", width=25, command=whatsapp_message)
whatsapp_button.grid(row=5 , column=1 ,padx=30 , pady=20)



button = tk.Button(root, text="Exit",width=25, command=on_exit)
button.grid(row=5 , column=2 ,padx=20 , pady=30)




# Create a menu bar
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

# Create a file menu
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=on_exit)
menu_bar.add_cascade(label="File", menu=file_menu)

# Create a help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
help_menu.add_command(label="Team Info", command=do_something)
menu_bar.add_cascade(label="About", menu=help_menu)



# Start the Tkinter event loop

root.mainloop()
