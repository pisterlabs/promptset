import openai
import os
import sys
import docx2txt
import PySimpleGUI as sg
import sounddevice as sd
from dotenv import load_dotenv
import threading

load_dotenv()
terminalOutputs = []

def GPT4():
    messages.append( {"role": "user", "content": message} )
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        top_p=topP,
        temperature=temp,                                                   
        frequency_penalty=0.0,                                              
        presence_penalty=0.0 )
    reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    print("\n" + reply + "\n")
    conversation.append(reply)
    conversationString = "\n \n".join(conversation)
    window["-OUTPUT-"].update(conversationString)


def run_GPT4_thread():
    # This function will run in a separate thread
    GPT4()

def whisper_transcript():
        strippedLocation = message.split("L>")[1].strip()
        audioFile = open(strippedLocation, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audioFile)
        transcript = str(transcript)
        
        #remove the {} from the start and end of the transcript
        finalTranscript = transcript[12:-1]
        message = "If possible, use this as a primary source for any further questions:" + finalTranscript

def browse_textfiles():
    sourceLocation = message.split("R>")[1].strip()
    with open(sourceLocation) as file:
        txtContent = file.read().replace('\n',' ')
    message = "If possible, use this as a primary source for any further questions:" + txtContent

##extra encouragement;)
messages = [
    {"role": "system", "content": "Hello, what can I do for you today?"},
    {"role": "user", "content": "You are a helpful assistant."},
    {"role": "system", "content": "Thank you, I try my best."}
]
conversation = []
system_msg = "You are an AI."
OPENAI_API_KEY_fromApp = None

OPENAI_API_KEY_fromEnv = os.getenv("OPENAI_API_KEY")

    

#customize the GUI
sg.theme("DarkGreen2")
fontT = ("Arial", 20)
fontB = ("Arial", 15)
fontS = ("Arial", 10)
menu = [["Settings", ["Change Temperature", "Change Top-P", "Change API Key"]], ["Debug", ["Show Terminal"]]]
title = sg.Text("To manually import a .txt file begin with R> \nTo manually import an audio file begin with L>", font=fontT)
folderlocation = os.path.dirname(os.path.realpath(__file__))
duck_img = sg.Image(f'{folderlocation}/duck.png', size=(65,65), enable_events=True, key="-DUCK-")
mic_button = sg.Button("Audio Input", key="-RECORD-")
send_button = sg.Button("Send")
clear_button = sg.Button("Clear")
quit_button = sg.Button("Quit")
internetsearch = sg.Checkbox("Search Internet", default=False, key="-SEARCH-")
mainInput = sg.Multiline("", size=(100,7), key="-INPUT-", font=fontB)
secondaryInput = sg.Input(enable_events=True, key='-IN-',font=fontS, expand_x=True), sg.FileBrowse()
mainOutput = sg.Multiline("", size=(100,24), key="-OUTPUT-", font=fontB)
TempAsk = 5
TopPAsk = 5

#define the layout of the GUI
layout = [[sg.Menu(menu)],
        [title, sg.Column([[duck_img]])],
        [mainOutput],
        [mainInput],
        [secondaryInput],
        [send_button, clear_button, quit_button]
    ]



# Create the GUI window
window = sg.Window("GPT-4", layout, size=(700, 700), finalize=True)

# Start the event loop
while True:
    
    event, values = window.read()
    if event == "Quit" or event == sg.WIN_CLOSED:
        break
    
    while True: 
        
        if OPENAI_API_KEY_fromEnv != None:
            openai.api_key = OPENAI_API_KEY_fromEnv
        elif OPENAI_API_KEY_fromApp != None:
            openai.api_key = OPENAI_API_KEY_fromApp
        elif OPENAI_API_KEY_fromEnv == None and OPENAI_API_KEY_fromApp == None:
            if getattr(sys, 'frozen', False):
                application_path = sys._MEIPASS
            else:
                application_path = os.path.dirname(os.path.abspath(__file__))
            openai.api_key_path = os.path.join(application_path, "APIKEY.txt")
            
        if event == "Change API Key":
            APIASK = sg.popup_get_text("Please enter your API Key", title="API Key")
            print("Key = ", APIASK)
            if APIASK != None:
                OPENAI_API_KEY_fromApp = APIASK
            break
        elif event == "Show Terminal":
            TerminalContent = '\n'.join(terminalOutputs)
            TerminalPopup = sg.popup_scrolled(TerminalContent + "this doesnt work yet, please help", size=(100, 100), title="Terminal", font=fontS)
            print("Terminal Opened")
            break
        elif event == "Change Temperature":
            TempAsk = sg.popup_get_text("Enter Temperature value between 1-10", title="Temperature")
            if TempAsk == None:
                TempAsk = 5
            print("Temperature Changed")
            break
        elif event == "Change Top-P":
            TopPAsk = sg.popup_get_text("Enter Top-P value between 1-10", title="Top-P")
            if TopPAsk == None:
                TopPAsk = 5
            print("Top-P Changed")
            break
        else:
            break
            
    
    docURL = values["-IN-"]
    if event == "-IN-":
        if ".mp3" in docURL:
            message = ("L>" + docURL)
        elif ".txt" in docURL:
            message = ("R>" + docURL)
        elif ".docx" in docURL:
            convFile = docx2txt.process(docURL)
            txtContentCV = convFile.replace('\n',' ')
            message = "If possible, use this as a primary source for any further questions:" + txtContentCV
            
    else:
        messagePreRep = values["-INPUT-"]
        message = messagePreRep.replace('\n',' ')
        
    if event == "Clear":
        window["-INPUT-"].update("")
        window["-OUTPUT-"].update("")
        conversation = []
        messages = []
        messages.append( {"role": "user", "content": "Hello, what can I do for you today?"} )
        messages.append( {"role": "system", "content": "You are a helpful assistant."} )
        messages.append( {"role": "user", "content": "Thank you, I try my best."} )
        continue
    
    temp = (float(TempAsk) * 0.1)
    topP = (float(TopPAsk) * 0.1)

    if event == "Send":
        conversation.append(values["-INPUT-"])
        threading.Thread(target=run_GPT4_thread).start()
        window["-IN-"].update("")
        window["-INPUT-"].update("")
        window.refresh()
    
    #OPEN TEXT FILES
    if "R>" in message:
        browse_textfiles()
    #TRANSCRIBE AUDIO INTO PROMPT
    if "L>" in message:
        whisper_transcript()

