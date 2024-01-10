import openai
import pyttsx3
import speech_recognition as sr
import requests
import smtplib
import re


# Set your OpenAI API key
openai.api_key = ""
OPENWEATHERMAP_API_KEY = ""

EMAIL_ADDRESS = "joshsoke@gmail.com"
EMAIL_PASSWORD = ""


# Initialize text-to-speech engine
tts_engine = pyttsx3.init()

voices = tts_engine.getProperty('voices')
for i, voice in enumerate(voices):
    print(f"Voice {i}: {voice.name}, {voice.languages}, {voice.gender}, {voice.age}")

def listen(expected_phrases=None):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=15) 
        
    try:
        text = recognizer.recognize_google(audio, show_all=False, language='en-US', preferred_phrases=expected_phrases)
        print(f"You said: {text}")

        if expected_phrases:
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            match = re.search(email_pattern, text)
            if match:
                print(f"Found email: {match.group()}")
                return match.group()

        return text
    except:
        print("Sorry, I couldn't understand.")
        return None


def is_email_trigger(text):
    email_triggers = [
        "send email",
        "compose email",
        "create email",
        "write email",
        "email someone",
    ]
    for trigger in email_triggers:
        if trigger in text.lower():
            return True
    return False

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",  # Change this to "gpt-3.5-turbo" when it becomes available
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        main = data["main"]
        weather_desc = data["weather"][0]["description"]
        temp = main["temp"]
        return f"The current weather in {city} is {weather_desc} with a temperature of {temp}°C."
    else:
        return f"Sorry, I couldn't get the weather for {city}."

def get_city_name():
    speak("Please tell me the name of the city.")
    city = listen()
    if city is None:
        city = "unknown"
    return city

def send_email(subject, body, to_email):
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(EMAIL_ADDRESS, to_email, message)
            return "Email sent successfully."
    except Exception as e:
        print(e)
        return "Failed to send email."
    
def speak(text):
    print(f"Chatbot: {text}")
    voices = tts_engine.getProperty('voices')
    # Change the voice index to choose a different voice
    voice_index = 0
    tts_engine.setProperty('voice', voices[voice_index].id)
    tts_engine.say(text)
    tts_engine.runAndWait()

def send_email(subject, body, to_email):
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            message = f"Subject: {subject}\n\n{body}"
            server.sendmail(EMAIL_ADDRESS, to_email, message)
            return "Email sent successfully."
    except Exception as e:
        print(e)
        return "Failed to send email."
    

if __name__ == "__main__":
    print("Welcome to the GPT-3.5 Turbo Chatbot!")
    print("Press Ctrl+C to quit.")
    prompt = "you are a helpful assistant and a master in various things such as science, tech and history"

    while True:
        try:
            user_input = listen()
            if user_input is not None:
                user_input = user_input.strip()
                if "weather" in user_input.lower():
                    city = get_city_name()
                    weather_info = get_weather(city, OPENWEATHERMAP_API_KEY)
                    print(weather_info)
                    speak(weather_info)
                elif is_email_trigger(user_input):
                    speak("Please tell me the email address.")
                    expected_email_domains = ["gmail.com", "yahoo.com", "hotmail.com", "fitzba.com", "outlook.com", "facebook.com"]
                    to_email = listen(expected_phrases=expected_email_domains)
                    speak("Please tell me the subject.")
                    subject = listen()
                    speak("Please tell me the content.")
                    body = listen()
                    response = send_email(subject, body, to_email)
                    print(response)
                    speak(response)
                else:
                    prompt += f"\nUser: {user_input}"
                    response = generate_response(prompt)
                    prompt += f"\nChatbot: {response}"
                    speak(response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
