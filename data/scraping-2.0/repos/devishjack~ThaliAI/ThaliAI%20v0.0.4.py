import openai
import speech_recognition as sr
import time
import elevenlabs
from elevenlabs import set_api_key
import os
from dotenv import load_dotenv
import gmaps
import gmaps.directions
from tkinter import *
from PIL import ImageTk, Image

listener = sr.Recognizer()
root = Tk()
root.title('ThaliAI')
root.iconbitmap('ThaliaExpressions\\ThaliAIIcon.ico')
my_img = ImageTk.PhotoImage(Image.open('ThaliaExpressions\\ThaliaNorm.png'))
my_label = Label(image=my_img)

happywords = ['happy','thrilled','excited','excitment']
sarcasticwords = ['sarcastic','sarcasm']

def start(client): 
     personality = os.getenv('Personality')
     voice = os.getenv('voice')
     assistant_key = os.getenv("assistantKey")
     name = os.getenv("name")
     if (voice == ""):
         os.environ["voice"] = set_voice()
         voice = os.getenv("voice")
     if (personality == ""):
         os.environ["personality"] = set_personality(voice)
         personality = os.getenv('Personality')
         print(personality)
     if (name == ""):
         os.environ["name"] = set_name(voice)
         name = os.getenv("name")
     if (assistant_key == ""):
         os.environ["assistantKey"] = get_assisstant_key(client, personality, voice, name)
         assistant_key = os.getenv('assistantKey')
         
         
def set_personality(voice):
    talk("what personality would you like me to have?", voice)
    command = take_command()
    return command

def set_voice():
    talk("choose the voice for your personal assistant", "Thalia")
    talk("do you like Thalia?", "Thalia")
    talk("or maybe Dorothy?", "Dorothy")
    talk("maybe something more masculine like Josh", "Josh")
    talk("or something else entirely?", "Fin")
    command = take_command()
    return command.title()

def set_name(voice):
    talk("what would you like my name to be?", voice)
    command = take_command()
    return command

def get_assisstant_key(client, personality, voice, name):
    talk("creating your personalized assistant now", voice)
    assistant = client.beta.assistants.create(name=name,model="gpt-3.5-turbo-1106",instructions=personality,tools=[{"type": "code_interpreter"}])
    return assistant.id
          

def configure():
    load_dotenv()

def get_direction(start, end):
    gmaps.configure(api_key=os.getenv('MapsAPI'))
    fig = gmaps.figure()
    layer = gmaps.directions.Directions(start,end,mode='driving',depature_time='now')
    fig.add_layer(layer) 



def take_command():
    command = ''
    with sr.Microphone() as source:
        print('listening...')
        voices = listener.listen(source)
        try:
         command = listener.recognize_google(voices, language = 'en-IN')
         command = command.lower()
        except Exception as e:
            print(e)
    return command

def change_image(emo):
    global my_img
    global my_label

    my_label.grid_forget()
    my_img = ImageTk.PhotoImage(Image.open('ThaliaExpressions\\' + emo + '.png'))
    my_label = Label(image=my_img)
    my_label.grid(row=0,column=0,columnspan=3)
    root.update()

def talk(text, voice):
    audio = elevenlabs.generate (
        text = text,
        voice = voice,
    )
    for word in happywords:
        if word in text:
            print('changing expression happy')
            change_image('ThaliaHappy')
            break
        else:
            print('changing expression norm')
            change_image('ThaliaNorm')
    for word in sarcasticwords:
        if word in text:
            print('changing expression annoyed')
            change_image('ThaliaAnnoyed')
            break
        else:
            print('changing expression norm')
            change_image('ThaliaNorm')
    elevenlabs.play(audio)

def generate_response(client, prompt, thread, assis_id):
    print("generating response...")
    message = client.beta.threads.messages.create(
       thread_id = thread.id,
       role = "user",
       content = prompt,
    )
    run = client.beta.threads.runs.create(
       thread_id = thread.id,
       assistant_id = assis_id,
    )
    time.sleep(5)
    runStatus = client.beta.threads.runs.retrieve(
       thread_id = thread.id,
       run_id = run.id,
    )
    if runStatus.status == "completed":
         messages = client.beta.threads.messages.list(
             thread_id = thread.id,
            )
         for mesg in (messages.data):
             print(mesg.role + ": " + mesg.content[0].text.value)
             return (mesg.content[0].text.value)

def main():
        my_label.grid(row=0,column=0,columnspan=3)
        root.update()
        configure()
        set_api_key(api_key=os.getenv('elevenAPI'))
        client = openai.OpenAI(api_key = os.getenv('OpenAPI'))
        thread = client.beta.threads.create()
        start(client)
        while True:
            command = take_command()
            if 'thalia' in command:
                talk('how can I help you?', os.getenv("voice"))
                while True:
                    command = take_command()
                    if 'quit' in command:
                        response = generate_response( client, command, thread, os.getenv("assistantKey"))
                        talk(response, os.getenv("voice"))
                        client.beta.threads.delete(thread.id)
                        quit()
                    else:
                        response = generate_response(client, command, thread, os.getenv("assistantKey"))
                        talk(response, os.getenv("voice"))
 
if __name__ == "__main__":
   main()

