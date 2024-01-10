import os
import speech_recognition as sr
import openai
import tkinter as tk
import threading
import time
from gtts import gTTS
from playsound import playsound
from tkinter import *
from tkinter import filedialog  # For getting the file path
from tkinter import simpledialog  # For getting the api key from the user

# Function to get the OpenAI API key from the user
def get_openai_api_key():
    if os.path.exists("api.txt"):
        # If the "api.txt" file exists, read the API key from it
        with open("api.txt", "r") as file:
            api_key = file.read().strip()
    else:
        # Prompt the user for the API key
        api_key = simpledialog.askstring("OpenAI API Key", "Please enter your OpenAI API key:")
        if api_key:
            # If the user provided an API key, save it to the "api.txt" file
            with open("api.txt", "w") as file:
                file.write(api_key)
        else:
            print("API Key not provided.")
    return api_key

# Prompt the user for the API key or read it from the "api.txt" file
api_key = get_openai_api_key()
if not api_key:
    exit()  # Exit the application if no API key is provided

# Set the OpenAI API key
openai.api_key = api_key

openai.organization = "org-SwXi7R63XlYwpVttB2KA1al8"
openai.Model.list()
r = sr.Recognizer()
m = sr.Microphone()
os.system("del wake.mp3")
os.system("del result.mp3")
myobj = gTTS("Yes?", lang='en', slow=False)
myobj.save("wake.mp3")

root = Tk()
root.title("AI Chat")
root.geometry("600x200+200+200")
root.configure(bg="black")
root.resizable(False, False)


#icon
image_icon = PhotoImage(file="C:/Users/mkivn/Desktop/Fall 2023/csc499/New folder/icon3.png")
root.iconphoto(False, image_icon) 

#top frame  
Top_frame = Frame(root, bg="white", width=600, height=100)
Top_frame.place(x=0, y=0)

# Load the original image
original_image = PhotoImage(file="C:/Users/mkivn/Desktop/Fall 2023/csc499/New folder/iconTop.png")

# Get the dimensions of the Top_frame
frame_width = 200
frame_height = 100

# Resize the image to match the frame dimensions
resized_image = original_image.subsample(original_image.width() // frame_width, original_image.height() // frame_height)

# Create a Label with the resized image
Label(Top_frame, image=resized_image, bg="white").place(x=0, y=0)
Label(Top_frame, text="CSC-499 AI Chat", font=("Arial 20 bold"), bg="white", fg="black").place(x=200, y=30)


# function to exit the program
def exit_program():
    global chat_terminate
    chat_terminate = True
    root.quit()

# exit button
exit_button = Button(root, text="Exit", command=exit_program)
exit_button.pack()
exit_button.place(x=550, y=150)
root.protocol("WM_DELETE_WINDOW", exit_program)#exit if user clicks the X

output_text = Text(root, height=5, width=60, wrap=WORD)
output_text.place(x=30, y=110)

def ask_chatGPT(messages, model="gpt-3.5-turbo"):
    response = openai.ChatCompletion.create(
        model=model, 
        messages=messages,
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].message.content
    messages.append(response.choices[0].message)
    return message

messages = []


chat_terminate = False
result_mp3 = "result.mp3"  # Initialize the result MP3 file name

def listen_for_wake_word():
    global chat_terminate
    global result_mp3  # Use the global file name

    while not chat_terminate:
        with m as source:
            audio = r.listen(source, phrase_time_limit=2)

        try:
            text = r.recognize_google(audio)
            wakeIndex = text.find('test')
            if wakeIndex >= 0:
                playsound("wake.mp3")
                print("Speak....")
                with m as source:
                    audio = r.listen(source, phrase_time_limit=5)
                    text = r.recognize_google(audio).lower()
                    prompt = text

                    # Use the same file name for the result MP3
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                    messages.append({"role": "user", "content": prompt})

                    gpt_response = ask_chatGPT(messages)
                    print(gpt_response)
                    output_text.insert(INSERT, "User: " + prompt + "\n")
                    output_text.insert(INSERT, "AI: " + gpt_response + "\n")
                    output_text.see(END)

                    myobj = gTTS(gpt_response, lang='en', slow=False)
                    myobj.save(result_mp3)
                    
                    playsound(result_mp3)
                    os.remove(result_mp3)
        except Exception as e:
            print("Sorry, couldn't hear you")
            print(e)




# Create a thread to listen for the wake word
wake_word_thread = threading.Thread(target=listen_for_wake_word)

# Start the wake word listening thread
wake_word_thread.start()

# Start the main Tkinter event loop
root.mainloop()