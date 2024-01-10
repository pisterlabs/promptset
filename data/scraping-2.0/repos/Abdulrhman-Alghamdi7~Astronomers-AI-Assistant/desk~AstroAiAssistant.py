import tkinter as tk
from datetime import datetime
import csv
import openai
from skyfield.api import load, Topos, utc
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()
configure()
api_key = os.getenv('api_key')

data = []

with open('/Users/abdulrhmanalghamdi/Library/Mobile Documents/com~apple~CloudDocs/programming💻/AstronomersAiAssistant/astronomicalEvents.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row)
def askfordistance(a, model="gpt-3.5-turbo", max_tokens=10):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "user", "content": f"give me just the distance from earth to {a} in 10 characters max"}
            ],
            max_tokens=max_tokens,
            api_key=api_key
        )

        if response.choices:
            return response.choices[0].message["content"].strip()
        else:
            return "No answer provided."
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def askforinfo(a, model="gpt-3.5-turbo", max_tokens=100):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "user", "content": f"give me a summary about {a} in 100 characters max"}
            ],
            max_tokens=max_tokens,
            api_key=api_key
        )

        if response.choices:
            return response.choices[0].message["content"].strip()
        else:
            return "No answer provided."

    except Exception as e:
        return f"An error occurred: {str(e)}"
r = []
def get_user_location():
    
    try:
        geolocator = Nominatim(user_agent="get_user_location")
        location = geolocator.geocode("")

        if location:
            return location.latitude, location.longitude
        else:
            r.append("\nUnable to determine your location. Using default location(0,0).")
            default_latitude = 0.0
            default_longitude = 0.0
            return default_latitude, default_longitude
    except GeocoderTimedOut:
        r.append("\nGeocoding service timed out. Unable to determine your location. Using default location(0,0).")
        default_latitude = 0.0
        default_longitude = 0.0
        return default_latitude, default_longitude

def get_celestial_body_info(body_name):
    planets = load('de421.bsp')

    object = planets[body_name]

    observer_location = get_user_location()

    if observer_location is not None:
        observer_latitude, observer_longitude = observer_location

        ts = load.timescale()
        time = ts.now()
        
        observer = Topos(observer_latitude, observer_longitude)

        observer_position = observer.at(time)
        object_position = object.at(time)

        separation = object_position.separation_from(observer_position)

        r.append(f'\nRight Ascension: {object_position.radec()[0].hours}\nDeclination: {object_position.radec()[1].degrees}\nseparation: {separation.degrees}')
        return f"Name: {body_name}\nAbout: {askforinfo(body_name)} {''.join(r)}"
    else:
        return None

def get_events_in_date_range(start_date, end_date):
    events_in_range = []
    start_date = datetime.strptime(start_date, '%Y/%m/%d')
    end_date = datetime.strptime(end_date, '%Y/%m/%d')
    for event in data[1:]:  # Skip the header row
        event_start_date = datetime.strptime(event[2].strip(), '%Y/%m/%d')
        event_end_date = datetime.strptime(event[3].strip(), '%Y/%m/%d')
        if start_date <= event_end_date and end_date >= event_start_date:
            events_in_range.append(event[1:])
    s = ''
    for i in events_in_range:
        s += f'Event name: {i[0]}\nStart date: {i[1]}\nEnd date: {i[2]}\nEvent description: {i[3]}\n\n'
    return s

def ask_gpt3_5_chat(question, model="gpt-3.5-turbo", max_tokens=500):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a chatbot."},
                {"role": "user", "content": question}
            ],
            max_tokens=max_tokens,
            api_key=api_key
        )

        if response.choices:
            return response.choices[0].message["content"].strip()
        else:
            return "No answer provided."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def chatelem():
    oblabel.pack_forget()
    objetb.pack_forget()
    sendbutobj.pack_forget()
    inflabel.pack_forget()
    infrep.pack_forget()
    evlabel.pack_forget()
    sdlabel.pack_forget()
    esdtb.pack_forget()
    enlabel.pack_forget()
    endtb.pack_forget()
    sendbuteve.pack_forget()
    infelabel.pack_forget()
    inferep.pack_forget()
    chlabel.pack()
    chattb.pack()
    sendbutt.pack()
    replabel.pack()
    chatrep.pack()
    
    
    
def eventselem():
    chattb.pack_forget()
    chatrep.pack_forget()
    sendbutt.pack_forget()
    chlabel.pack_forget()
    replabel.pack_forget()
    oblabel.pack_forget()
    objetb.pack_forget()
    sendbutobj.pack_forget()
    inflabel.pack_forget()
    infrep.pack_forget()
    evlabel.pack()
    sdlabel.pack()
    esdtb.pack()
    enlabel.pack()
    endtb.pack()
    sendbuteve.pack()
    infelabel.pack()
    inferep.pack()

def objectelem():
    chattb.pack_forget()
    chatrep.pack_forget()
    sendbutt.pack_forget()
    chlabel.pack_forget()
    replabel.pack_forget()
    evlabel.pack_forget()
    sdlabel.pack_forget()
    esdtb.pack_forget()
    enlabel.pack_forget()
    endtb.pack_forget()
    sendbuteve.pack_forget()
    infelabel.pack_forget()
    inferep.pack_forget()
    oblabel.pack()
    objetb.pack()
    sendbutobj.pack()
    inflabel.pack()
    infrep.pack()

def sendai():
    q = chattb.get("1.0", "end-1c")
    prompt = f"consider that you are an astronomer who can answer any astronomical or scientific question, but you cannot answer any non-astronomical or scientific question. You answer with (this is outside the scope of my knowledge). Based on the previous information, answer the following question: {q} within 500 characters."
    chatrep.delete("1.0", "end-1c")
    a = ask_gpt3_5_chat(prompt)
    chatrep.insert("end", a)
    chattb.delete("1.0", "end-1c")

def srhevnt():
    sd = esdtb.get("1.0", "end-1c")
    ed = endtb.get("1.0", "end-1c")
    inferep.delete("1.0", "end-1c")
    a = get_events_in_date_range(sd,ed)
    inferep.insert("end", a)
    esdtb.delete("1.0", "end-1c")
    endtb.delete("1.0", "end-1c")

def srhobj():
    obj = objetb.get("1.0","end-1c")
    a = get_celestial_body_info(obj)
    infrep.delete("1.0", "end-1c")
    infrep.insert("end", a)
    objetb.delete("1.0", "end-1c")

root = tk.Tk()
root.geometry("800x800")
root.title('Astronomers AI Assistant')
#chat
chlabel = tk.Label(root, text="Enter your Question")
chattb = tk.Text(root, height=2, width=100)
sendbutt = tk.Button(root, text="Send",command=sendai)
replabel = tk.Label(root, text="Replay")
chatrep = tk.Text(root, width=100, height=40)
#objects
oblabel = tk.Label(root, text="Enter objects scientific name")
objetb = tk.Text(root, height=2, width=100)
sendbutobj = tk.Button(root, text="Search",command=srhobj)
inflabel = tk.Label(root, text="Object info")
infrep = tk.Text(root, width=100, height=40)
#events
evlabel = tk.Label(root, text="Search for Astronomical events")
sdlabel = tk.Label(root, text="Start date")
esdtb = tk.Text(root,height=2, width=20)
enlabel = tk.Label(root, text="End date")
endtb = tk.Text(root,height=2, width=20)
sendbuteve = tk.Button(root, text="Search",command=srhevnt)
infelabel = tk.Label(root, text="Event info")
inferep = tk.Text(root, width=100, height=35)

label1 = tk.Label(root, text="Welcome to the Astronomers AI Assistant")
label1.pack()

button_chat = tk.Button(root, text="Chat with AI", command=chatelem)
button_chat.pack()

button_event = tk.Button(root, text="Saerch for Event", command=eventselem)
button_event.pack()

button_object = tk.Button(root, text="Search for Object", command=objectelem)
button_object.pack()

root.mainloop()
