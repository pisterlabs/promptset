import speech_recognition as sr
import os
import webbrowser
import openai
import datetime
import random
import pyttsx3
import requests
import mysql.connector  # Import the MySQL Connector
import pprint
import google.generativeai as palm

api_key= "Enter your API key here"
mysql_config = {
    'user': 'Enter your username here',  # Your MySQL username
    'password': 'Enter your password here',  # Your MySQL password
    'host': 'Enter host name',  # Example: 'localhost'
    'database': 'Enter database name',  # Your MySQL database name
}
chat_history_table = "conversations"  # MySQL table to store chat history

conn = mysql.connector.connect(**mysql_config)
cursor = conn.cursor()

# Create the table to store chat history if it doesn't exist
create_table_query = f'''CREATE TABLE IF NOT EXISTS {chat_history_table}
                         (user_input TEXT, bot_response TEXT)'''
cursor.execute(create_table_query)
conn.commit()
listening = False  # Variable to track if the chatbot is listening or not
def save_chat(user_input, bot_response):
    # Save the conversation to the MySQL database
    insert_query = f"INSERT INTO {chat_history_table} (user_input, bot_response) VALUES (%s, %s)"
    cursor.execute(insert_query, (user_input, bot_response))
    conn.commit()


def reset_chat():
    # Delete all chat history from the MySQL database
    delete_query = f"DELETE FROM {chat_history_table}"
    cursor.execute(delete_query)
    conn.commit()

def retrieve_chat():
    # Retrieve all conversations from the MySQL database
    select_query = f"SELECT * FROM {chat_history_table}"
    cursor.execute(select_query)
    return cursor.fetchall()


def chat(query):
    response = get_bot_response(query)
    say(response)
    save_chat(query, response)
    return response


def get_bot_response(user_input):
    # Retrieve previous conversations
    conversations = retrieve_chat()

    # Append previous user inputs to prompt
    prompt = "\n".join([f"User: {input}" for input, _ in conversations])

    # Append previous bot responses to prompt
    prompt += "\n".join([f"Luna: {response}" for _, response in conversations])

    prompt += f"\nUser: {user_input}\nLuna: "

    # Generate bot response using OpenAI
    openai.api_key = api_key
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    bot_response = response["choices"][0]["text"]

    return bot_response


def ai(prompt):
    openai.api_key = api_key
    text = f"OpenAI response for Prompt: {prompt} \n *************************\n\n"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    text += response["choices"][0]["text"]

    if not os.path.exists("Openai"):
        os.mkdir("Openai")

    with open(f"Openai/{''.join(prompt.split('intelligence')[1:]).strip()}.txt", "w") as f:
        f.write(text)


def say(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    # Change the index to use a different voice
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()


def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language="en-in")
            #query = r.recognize_google(audio, language="hi-in")
            print(f"User said: {query}")
            return query
        except Exception as e:
            return "Some Error Occurred. Sorry from Luna"


def get_weather(city):
    url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}"
    response = requests.get(url)
    data = response.json()
    weather = data["current"]["condition"]["text"]
    temperature = data["current"]["temp_c"]
    return f"The weather in {city} is {weather} with a temperature of {temperature}°C."


if __name__ == '__main__':
    print('Welcome To Luna A.I')
    say("Hello, soumya")
    is_active = True  # Variable to control the conversation loop
    while is_active:
        print("Listening...")
        query = takeCommand().lower()
        sites = [["youtube", "https://www.youtube.com"], ["wikipedia", "https://www.wikipedia.com"],
                 ["google", "https://www.google.com"],["chat GPT", "https://chat.openai.com"], ["youtube music", "https://music.youtube.com/"],
                ]
        for site in sites:
            if f"open {site[0]}" in query:
                say(f"Opening {site[0]} sir...")
                webbrowser.open(site[1])

        if "open music" in query:
            musicPath = r"C:\Users\SOUMYA\Music"
            os.startfile(musicPath)
        if "open android studio" in query:
            androidPath = r"C:\PerfLogs\android studio\bin\studio64.exe"
            os.startfile(androidPath)

        elif "the time" in query:
            now = datetime.datetime.now()
            time = now.strftime("%I:%M %p")
            say(f"Sir, the time is {time}.")

        elif "weather" in query:
            city = "Bhubaneswar"  # Replace with user-provided city or implement speech recognition
            weather_info = get_weather(city)
            say(weather_info)

        elif "open calculator" in query:
            # Modify the path to the desired application or remove this block if not applicable
            calculator = r"C:\Windows\System32\calc.exe"
            os.startfile(calculator)

        elif "open notepad" in query:
            # Modify the path to the desired application or remove this block if not applicable
            notepad = r"C:\Windows\System32\notepad.exe"
            os.startfile(notepad)

        elif "open whatsapp" in query:

             whatsappPath = r"C:\Path\to\WhatsApp.exe"

             os.startfile(whatsappPath)


        elif "open mail" in query:

             mailPath = r"C:\Path\to\Mail.exe"

             os.startfile(mailPath)


        elif "open file explorer" in query:

            explorerPath = r"C:\Windows\explorer.exe"

            os.startfile(explorerPath)


        elif "open microsoft store" in query:

            storePath = r"C:\Windows\System32\ms-store://"

            os.startfile(storePath)


        elif "open settings" in query:

            settingsPath = r"C:\Windows\System32\control.exe"

            os.startfile(settingsPath)
        elif "open cmd" in query:
            # Modify the path to the desired application or remove this block if not applicable
            cmdpath = r"C:\Windows\System32\cmd.exe"
            os.startfile(cmdpath)
        elif "open control panel" in query:
            # Modify the path to the desired application or remove this block if not applicable
            controlePath = r"C:\Windows\System32\control.exe"
            os.startfile(controlePath)
        elif "open task manager" in query:
            # Modify the path to the desired application or remove this block if not applicable
            taskPath = r"C:\Windows\System32\Taskmgr.exe"
            os.startfile(taskPath)
        elif "open voice access" in query:
            # Modify the path to the desired application or remove this block if not applicable
            voicePath = r"C:\Windows\System32\VoiceAccess.exe"
            os.startfile(voicePath)

        elif "translate" in query:
            api_key = "YOUR_TRANSLATION_API_KEY"
            say("Sure, what would you like to translate?")
            text_to_translate = takeCommand().lower()  # Capture user input for text to translate
            say("Great! Which language would you like to translate it to?")
            target_language = takeCommand().lower()  # Capture user input for target language
            url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}"
            params = {
                "q": text_to_translate,
                "target": target_language
            }
            response = requests.post(url, params=params)
            data = response.json()
            translated_text = data["data"]["translations"][0]["translatedText"]
            say(f"The translation is: {translated_text}")

        elif "using artificial intelligence" in query:
            ai(prompt=query)

        elif "stop" in query:
            say("Luna is now inactive.")
            is_active = False  # Set the conversation loop variable to False to exit the loop

        elif "luna" in query:
            print("Luna is now active.")

        elif "reset chat" in query:
            reset_chat()
            say("Chat history has been reset.")

        elif "restore chat" in query:
            conversations = retrieve_chat()
            if len(conversations) == 0:
                say("No chat history found.")
            else:
                say("Chat history:")
                for user_input, bot_response in conversations:
                    print(f"User: {user_input}")
                    print(f"Luna: {bot_response}")
                    print("------------------------")

        elif "shutdown" in query:
            say("Are you sure you want to shut down your computer?")
            confirm = takeCommand().lower()

            if "yes" in confirm:
                say("Shutting down your computer...Bye Boss")
                os.system("shutdown /s /t 0")
            else:
                say("Shutdown canceled.")
        else:
            print("Chatting...")
            chat(query)
