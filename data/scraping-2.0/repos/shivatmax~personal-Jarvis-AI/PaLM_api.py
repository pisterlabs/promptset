import re
import pyttsx3
import speech_recognition as sr
import webbrowser
import os
import datetime
import requests
import json
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.chains import RetrievalQA

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm




chatstr = ''

def pdf(prompt):
    load_dotenv()
    files_path = "files\Resume_Shiv.pdf"
    loaders = [UnstructuredPDFLoader(files_path)]

    # if "index" not in st.session:
    index = VectorstoreIndexCreator(
        embedding=GooglePalmEmbeddings(),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=0),
    ).from_loaders(loaders)

    llm = GooglePalm(temperature=0.1)  # OpenAI()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        return_source_documents=True,
    )


    query = prompt

    if query:
        response = chain(query)
        print("Jarvis: " + response["result"])
        say(response["result"])

def weather(location):
    base_url = "http://api.weatherapi.com/v1"
    api_key = "cd5d5d6f93004fb1886163157232709"
    url = f"{base_url}/current.json?key={api_key}&q={location} india&aqi=no"

    response = requests.get(url)
    if response.status_code == 200:
      data = response.json()
      temp_c = str(data['current']['temp_c'])
      condition = str(data['current']['condition']['text'])
      print(temp_c, condition, location)
      say("tempurature is " + temp_c + "celsius and condition is " + condition + "in "+location)
    else:
      print("Error:", response.status_code)
def chat(prompt):
    global chatstr

    chatstr += f"Shiv: {prompt}\n Jarvis: "
    
    url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=AIzaSyAmMtufl_TPyWdLbuCFD5Lh8IEpJWrEXYo'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
    "prompt": {
        "text": chatstr
    }
}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        response_json = response.json()['candidates'][0]['output']
        chatstr += f"{response_json}\n"
        say(response_json)
        return response_json
    except requests.exceptions.RequestException as err:
        print(f"An error occurred: {err}")
        return None

def ai(prompt):
    text = f"Palm ai response for prompt: {prompt} \n ************************************************\n\n"

    url = 'https://generativelanguage.googleapis.com/v1beta2/models/text-bison-001:generateText?key=AIzaSyAmMtufl_TPyWdLbuCFD5Lh8IEpJWrEXYo'
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
    "prompt": {
        "text": prompt
    }
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    try:
      response.raise_for_status()
      print(response.json())
    except requests.exceptions.HTTPError as err:
      print(f"HTTP error occurred: {err}")
    except requests.exceptions.JSONDecodeError as err:
      print(f"Error decoding JSON response: {err}")
    response_json = response.json()['candidates'][0]['output']

    text += response_json
    if not os.path.exists("Openai"):
        os.mkdir("Openai")
    
    with open(f"Openai/{prompt}.txt", "w") as f:
        f.write(text)

def say(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


def takeCommand():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in' or 'hi-in')
        print(f"You: {query}\n")

    except Exception as e:
        print(e)
        print("Say that again please...")
        return "None"
    return query


if __name__ == '__main__':
    say("Hey   i    am    jarvis")
    while True:
        query = takeCommand().lower()
        sites = ["google", "facebook", "instagram", "github","notion"]
        files = ["Music", "Videos","Downloads","Documents","Screenshots"]
        apps = ["spotify","brave"]
        for site in sites:
            if "open "+site in query:
             webbrowser.open("www." + site + ".com")
             say("Opening " + site + "sir")
        for file in files:
            if "open " + file.lower() in query:
                path = r"C:\Users\awast\\" + file
                os.startfile(path)
                say("Opening " + file + "sir")
        for app in apps:
            if "open " + app.lower() in query:
                os.startfile(app+".exe")
                say("Opening " + app + "sir")
        if "the time" in query:
            strTime = datetime.datetime.now().strftime("%H:%M")
            say(f"Sir the time is {strTime}")
        elif "open youtube" in query and "search" in query:
            search_query = query.replace("open youtube and search", "").strip()
            search_url = f"https://www.youtube.com/results?search_query={search_query}"
            webbrowser.open(search_url)
        elif "using ai" in query.lower():
            query = query.replace("using ai", "")
            ai(prompt=query) 
        elif "exit" in query:
            print("Bye sir")
            say("Bye sir")
            exit()
        elif "bye" in query:
            print("Bye sir")
            say("Bye sir")
            exit()
        elif "clear chat" in query:
            chatstr = ''
            print("Chat resetted")
            say("Chat resetted")
        elif "weather" in query:
            match = re.search(r'in\s+(\w+)', query, re.IGNORECASE)
            if match:
              location = match.group(1)
            else:
              location = "Delhi"
            weather(location)
        elif "tell me" in query:
            query = query.replace("tell me", "")
            pdf(prompt=query)
        else:
            print("chatting...")
            print("jarvis: ",chat(prompt=query))
            
