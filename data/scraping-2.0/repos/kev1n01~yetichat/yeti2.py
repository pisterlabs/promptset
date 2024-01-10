import openai
import speech_recognition as sr
from queue import Queue
from text_to_speech import pytts,tts
from task import exec_task
import os
from  dotenv import load_dotenv

#load the environment variables
load_dotenv()
#set the openai api key
openai.api_key = os.getenv('API_KEY') 

#declare the queue for the transcript of the microphone input and the default text chat message
transcript_queue = Queue()

#function to record the microphone input
#and put the transcript in the queue
def record_microphone(audio = None):
    r, mic = sr.Recognizer(), sr.Microphone()
    with mic as source:
        print("Escuchando...")
        try:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            text = r.recognize_google(audio, language='es-ES')
            #put the transcript in the queue
            transcript_queue.put(text)  
            #print the transcript
            print("User: ", text, end="\r\n")
        except:
            #if the microphone input is empty
            transcript_queue.put("")
#function to start the conversation with the Yeti
def start_conversation():
    active: bool = False #flag to know if the Yeti is active
    sleep: bool = False #flag to know if the Yeti is sleeping
    task_mode: bool = False #flag to know if the Yeti is in task mode
    #loop to record the microphone input
    while True:
        #if the Yeti is active and not in task mode
        if not task_mode:
            record_microphone()
            transcript_result = transcript_queue.get()

        #condition to active Yeti
        if not sleep and not active and transcript_result == "yeti":
            active = True #set the flag to active
            pytts(os.getenv('text_chat_default')) #say the text chat message default
            print("AI: ", os.getenv('text_chat_default'), end="\r\n")
            continue

        #condition to active task mode
        if not task_mode and transcript_result == "modo tarea":
            pytts(os.getenv('text_chat_mode_task')) #say the text mode task message
            task_mode = True #set the flag to active task mode
            continue

        #condition to interact with the Yeti in task mode
        if active and task_mode:
            task = exec_task() #execute the task
            print(task) 
            if task == 'Done':
                pytts('Tarea completada')
            if task == 'Exit':
                pytts(os.getenv('text_chat_not_mode_task')) #say the text not mode task message
                task_mode = False
            continue
        
        #condition to say the Yeti susurrate
        if transcript_result == "yeti callate" or transcript_result == "yeti cállate":
            #set decrement the volume and rate of the voice Yeti
            pytts("Entendido", 0.5, 100) 
            continue

        #condition to say the Yeti talk normal
        if transcript_result == "yeti no entiendo" or transcript_result == "no entiendo":
            #set increment the volume and rate of the voice Yeti
            pytts("¿Y ahora si me entiendes mejor?", 1.0, 140)
            continue

        #condition to reactivate the Yeti
        if sleep and transcript_result == "yeti":
            active = True
            sleep = False
            pytts(os.getenv('text_chat_after_sleep'))
            print("\nAI: ", os.getenv('text_chat_after_sleep'), end="\r\n")
            continue

        #condition to sleep the Yeti 
        if active and transcript_result == "yeti adios" or transcript_result == 'yeti adiós' or transcript_result == "yeti apagate" or transcript_result == "yeti apágate":
            pytts(os.getenv('text_chat_before_sleep'))
            print("\nAI: ", os.getenv('text_chat_before_sleep'), end="\r\n")
            active = False
            sleep = True
            continue
        #condition to yeti answer according to openai
        if not task_mode and not sleep and active and transcript_result != "":
            #call the openai api to get the answer
            response = openai.ChatCompletion.create(
                model = os.getenv('MODEL'),
                messages = [
                    {
                        "role": "system",
                        'content': os.getenv('CONTENT_SYSTEM') #set the content of the system 
                    },
                    {
                        "role": "user",
                        "content": transcript_result #set the question of the user
                    }
                ],
            )
            text = response.choices[0]['message']['content'] #get the answer of the openai api
            print("\nAI: ", text, end="\r\n") #print the answer
            pytts(text) #say the answer
            continue

#call the function to start the conversation
start_conversation()
