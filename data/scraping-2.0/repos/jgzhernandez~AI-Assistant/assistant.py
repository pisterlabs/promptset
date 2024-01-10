from tkinter import *
from tkinter import messagebox
from ai_assistant import Assistant
import threading
import os
import json
import openai

# Creata a UI using tkinter
root = Tk()
root.title("Mombot")
root.configure(bg="black")

# Center Window
width = 400
height = 500
root.config(width=width, height=height)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width / 2) - (width / 2)
y = (screen_height / 2) - (height / 2)
root.geometry('%dx%d+%d+%d' % (width, height, x, y))


# Create a frame if user has no api key.
def no_api_frame():
    
    # Delete all elements if the app started from the default frame
    for slave in root.grid_slaves():
        slave.destroy()

    def save_api():
        api_dict = {"api_key": api_key_entry.get()}
        with open('api.json', 'w') as api_file:
            json.dump(api_dict, api_file)
        
        # Test if the api is valid, if not ask for another one.
        try:
            with open('api.json') as api:
                key = json.load(api)
                openai.api_key = key['api_key']
            openai.Model.list()
            default_frame()
        except openai.error.AuthenticationError:
            messagebox.showerror("Error", "Please use a valid API key")
            no_api_frame()


    # Create Title
    title = Label(text="🤖MOMBOT🤖", background="black", foreground="white", font="Courier 40 bold")
    title.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 100), sticky="NEW")
    
    # Create a label for API KEY
    api_key = Label(text="API Key", background="black", foreground="white", font="Courier 20 bold")
    api_key.grid(row=1, column=1, columnspan=1, padx=10, pady=10, sticky="NSEW")

    # Create an entry for the API Key
    api_key_entry = Entry(root, width=30, borderwidth=0)
    api_key_entry.grid(row=2, column=1, columnspan=1, padx=10, pady=5, sticky="NSEW")

    # Create a submit button
    submit_button = Button(text="Submit", command=lambda: [save_api()], background="black",
                           foreground="white", font="Courier 10 bold")
    submit_button.grid(row=3, column=1, padx=10, pady=5, sticky="NSEW")

    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)


# Create the default frame of the app
def default_frame():
    assistant = Assistant()
    # Delete all elements if the app started from the no api frame
    for slave in root.grid_slaves():
        slave.destroy()

    # Create Title
    title = Label(text="MOMBOT", background="black", foreground="white", font="Courier 40 bold")
    title.grid(row=0, column=0, columnspan=3, padx=10, pady=(10, 10), sticky="NEW")

    # Create the logo "label"
    logo = Label(text="🤖", background="black", foreground="white", font="Courier 240 bold")
    logo.grid(row=1, column=1, columnspan=1, padx=10, pady=10, sticky="NSEW")

    # Center all elements
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)
    root.grid_columnconfigure(2, weight=1)

    # Create a thread to start the assistant
    def start_assistant():
        assistant.is_on = True
        assistant.voice_recognize()

    # Create another thread to modify the UI while the assistant is running
    def dynamic_ui():
        while True:
            if assistant.is_closing:
                break
            elif assistant.is_transcribing:
                title.config(text="MOMBOT\nis transcribing...", font="Courier 20 bold")
                logo.config(foreground="green")
            elif assistant.is_listening:
                title.config(text="MOMBOT\nis listening...", font="Courier 20 bold")
                logo.config(foreground="#ff007c")
            else:
                title.config(text="MOMBOT", font="Courier 40 bold")
                logo.config(foreground="white")

        # Exit the app after the sleep word
        os._exit(0)

    # Run the threads
    assistant_thread = threading.Thread(target=start_assistant)
    assistant_thread.start()
    dynamic_ui_thread = threading.Thread(target=dynamic_ui)
    dynamic_ui_thread.start()

# Check if user has an API key file then load the appropriate frame.
try:
    with open('api.json') as api:
        default_frame()
except FileNotFoundError:
    no_api_frame()

# Create a thread to run the UI
root.mainloop()
