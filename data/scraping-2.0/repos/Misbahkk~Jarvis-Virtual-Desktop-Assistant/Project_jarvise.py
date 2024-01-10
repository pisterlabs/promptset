import speech_recognition as sr
import win32com.client
import os
import webbrowser
import openai
from config import apikey
from config import weather_apikey
import datetime as dt
import requests




# find the weather using API 
def kel_to_cel_fahre(kelvin):
    celsius = kelvin - 273.15
    fahrenheit = celsius *1.8+32
    return celsius , fahrenheit
def weather_info(city):
    # city_name = "matli"
    data = requests.get("https://api.openweathermap.org/data/2.5/weather?q="+ city +"&appid="+weather_apikey).json()


    temp_kelvin = data['main']['temp']
    temp_celsius , temp_fahrenheit = kel_to_cel_fahre(temp_kelvin)
    feels_like_kelvin = data['main']['temp']
    feels_like_celsius , feels_like_fahrenheit = kel_to_cel_fahre(feels_like_kelvin)
    wind_speed = data['wind']['speed']
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']
    sunrise_time = dt.datetime.utcfromtimestamp(data['sys']['sunrise'] + data['timezone'])
    sunset_time = dt.datetime.utcfromtimestamp(data['sys']['sunset'] + data['timezone'])
    say(f"Temprature in {city}  : {temp_celsius:.2f}C or {temp_fahrenheit:.2f}F")
    print(f"Temprature in {city} feels like  : {feels_like_celsius:.2f}C or {feels_like_fahrenheit:.2f}F")
    print(f"Humidity in {city} : {humidity}")
    print(f"Wind Speed in {city} : {wind_speed}m/s")
    print(f"Gernal Weather in {city} : {description}")
    print(f"Sun rises in {city} at {sunrise_time} local time.")
    print(f"Sun sets in {city} at {sunset_time} local time.")

# Chat with jarvis like a chatgpt 
chatStr = ""


def chat(query):
    global chatStr
    openai.api_key = apikey
    chatStr += f"Misabh : {query}\n Jarvis : "

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": chatStr}
        ],
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message_content = response["choices"][0]["message"]["content"]
    say(message_content)
    chatStr += f"{message_content}\n"
    print(chatStr)
    return message_content




# when user day using ai then this part do work and crete another file and save date 
def ai(message):
    openai.api_key = apikey
    text = f"OpenIA response for Message : {message} \n*************************\n\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    message_content = response["choices"][0]["message"]["content"]
    # print(message_content)
    text += message_content
    if not os.path.exists("openai"):
        os.mkdir("openai")
    with open(f"openai/{''.join(message.split('AI')[1:]).strip()}.txt", "w") as f:
        f.write(text)


# say funtion 


def say(text):
    speak = win32com.client.Dispatch("SAPI.SpVoice")
    speak.Speak(text)

def wishMe():
    '''
    is function ma jarvis ham ko ganton ka hisab sa wish kare ga
    matlb ka morning or evening ka type saa
    '''
    hour = int(dt.datetime.now().hour)
    if hour >= 0 and hour < 12:
        say("GoOd Morning! Misbah")
    elif hour >= 12 and hour < 18:
        say("Good AfterNoon! Misbah")
    else:
        say("Good Evening! Misbah")
    say("I am jarvis! How can i Help You Mam")




# Take a command from user
def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1
        audio = r.listen(source)
        try:
            print("regonition......")
            query = r.recognize_google(audio, language="en-pk")
            print(f"User Said:{query}")
            return query
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError:
            return "Sorry, there was an error connecting to the speech recognition service."
        except Exception as e:
            return "SOme Error Occurred. Sorry from Jarvis"


if __name__ == '__main__':
    wishMe()
    while True:
        print("listening....")
        query = takeCommand()
        if "open" in query:
            sites = [["youtube", "https://youtube.com"], ["googel", "https://googel.com"],
                    ["wikipedia", "https://wikipedia.com"], ["facebook", "https://facebook.com"],]
            for site in sites:
                if f"Open {site[0]}".lower() in query.lower():
                    say(f"Opening {site[0]} sir....")
                    webbrowser.open(site[1])
        elif "open music" in query:
            musicPath = "C:/Users/ADMIN/Downloads/downfall-21371.mp3"
            os.startfile(musicPath)
        elif "the time" in query:
            hour = datetime.datetime.now().strftime("%H")
            min = datetime.datetime.now().strftime("%M")
            say(f"Mam time is {hour} and {min} minutes")

        elif "Using AI".lower() in query.lower():
            ai(message=query)
        elif "weather".lower() in query.lower():
            cities = ["Ahmadpur East"," Ahmed Nager Chatha"," Ali Khan Abad"," Alipur"," Arifwala"," Attock"," Bhera"," Bhalwal"," Bahawalnagar"," Bahawalpur"," Bhakkar"," Burewala"," Chillianwala"," Choa Saidanshah"," Chakwal"," Chak Jhumra"," Chichawatni"," Chiniot"," Chishtian"," Chunian"," Dajkot"," Daska"," Davispur"," Darya Khan"," Dera Ghazi Khan"," Dhaular"," Dina"," Dinga"," Dhudial Chakwal"," Dipalpur"," Faisalabad"," Fateh Jang"," Ghakhar Mandi"," Gojra"," Gujranwala"," Gujrat"," Gujar Khan"," Harappa"," Hafizabad"," Haroonabad"," Hasilpur"," Haveli Lakha"," Jalalpur Jattan"," Jampur"," Jaranwala"," Jhang"," Jhelum"," Kallar Syedan"," Kalabagh"," Karor Lal Esan"," Kasur"," Kamalia"," Kāmoke"," Khanewal"," Khanpur"," Khanqah Sharif"," Kharian"," Khushab"," Kot Adu"," Jauharabad"," Lahore"," Islamabad"," Lalamusa"," Layyah"," Lawa Chakwal"," Liaquat Pur"," Lodhran"," Malakwal"," Mamoori"," Mailsi"," Mandi Bahauddin"," Mian Channu"," Mianwali"," Miani"," Multan"," Murree"," Muridke"," Mianwali Bangla"," Muzaffargarh"," Narowal"," Nankana Sahib"," Okara"," Renala Khurd"," Pakpattan"," Pattoki"," Pindi Bhattian"," Pind Dadan Khan"," Pir Mahal"," Qaimpur"," Qila Didar Singh"," Rabwah"," Raiwind"," Rajanpur"," Rahim Yar Khan"," Rawalpindi"," Sadiqabad"," Sagri"," Sahiwal"," Sambrial"," Samundri"," Sangla Hill"," Sarai Alamgir"," Sargodha"," Shakargarh"," Sheikhupura"," Shujaabad"," Sialkot"," Sohawa"," Soianwala"," Siranwali"," Tandlianwala"," Talagang"," Taxila"," Toba Tek Singh"," Vehari"," Wah Cantonment"," Wazirabad"," Yazman"," Zafarwal",]
            for city in cities:
                if f"{city[0]}".lower() in query.lower():
                    weather_info(city)
        elif "Jarvis Quit".lower() in query.lower():
            exit()
        elif "Reset the chat".lower() in query.lower():
            chatStr = ""
        else:
            chat(query)
