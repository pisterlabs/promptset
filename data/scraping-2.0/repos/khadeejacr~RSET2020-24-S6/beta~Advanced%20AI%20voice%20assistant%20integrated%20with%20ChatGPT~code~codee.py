import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import json
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import sys
import openai

engine = pyttsx3.init()

openai.api_key = "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"

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        transcription = recognizer.recognize_google(audio)
        return transcription.lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't understand.")
    except sr.RequestError as e:
        print(f"Error occurred during transcription: {e}")

    return None

def add_task(task, tasks):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for i, (stored_task, timestamp) in enumerate(tasks):
        if task == stored_task:
            tasks[i] = (task, current_time)  # Update the existing task with a new timestamp
            break
    else:
        tasks.append((task, current_time))
    save_tasks(tasks)  # Save tasks to file
    print(f"Task '{task}' added at {current_time}")
    speak_text(f"Task '{task}' added at {current_time}")

def check_task(task, tasks):
    for stored_task, timestamp in tasks:
        if task == stored_task:
            print(f"Task '{task}' was done at {timestamp}")
            speak_text(f"Task '{task}' was done at {timestamp}")
            return

    print(f"Task '{task}' was not done")
    speak_text(f"Task '{task}' was not done")

def save_tasks(tasks):
    with open("tasks.txt", "w") as file:
        for task, timestamp in tasks:
            file.write(f"{task},{timestamp}\n")

def load_tasks():
    tasks = []
    try:
        with open("tasks.txt", "r") as file:
            for line in file:
                task, timestamp = line.strip().split(",")
                tasks.append((task, timestamp))
    except FileNotFoundError:
        print("No tasks found.")
    return tasks

def reminder():
    tasks = load_tasks()  # Load tasks from file

    while True:
        # speak_text("How can I assist you?")
        user_input = record_audio()

        if user_input:
            if "add task" in user_input:
                speak_text("What task did you complete?")
                task_input = record_audio()

                if task_input:
                    add_task(task_input, tasks)

            elif "check task" in user_input:
                speak_text("Which task would you like to check?")
                task_input = record_audio()

                if task_input:
                    check_task(task_input, tasks)

            elif "goodbye" in user_input:
                speak_text("Goodbye!")
                break


def save_text_as_audio(response, audio_f):
    engine.save_to_file(response, audio_f)
    engine.runAndWait()


def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
    except sr.RequestError:
        print("Could not request results from speech recognition service")
    return ""


def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=4000,
        stop=None,
    )
    return response.choices[0].text




def write_output_to_file(text, filename):
    with open(filename, "w") as file:
        file.write(text + "\n")


def VA():
    while True:
        print("Say 'Julia' to ask questions")
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio)
            if transcription.lower() == "julia":
                filename = "C:\\Users\\govind kiran\\Desktop\\project\\output.txt"
                audio_f = "C:\\Users\\govind kiran\\Desktop\\project\\audio.mp3"
                print("Ask a question")
                with sr.Microphone() as source:
                    print("Listening...")
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)

                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())

                text = transcribe_audio_to_text(filename)
                if text:
                    print(f"You said: {text}")
                    if text.lower().startswith("write"):
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
                        write_output_to_file(response, filename)
                    elif text.lower().startswith("record"):
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
                        save_text_as_audio(response, audio_f)
                    elif text.lower().startswith("rap"):
                        text="I want you to act as a rapper. You will come up with powerful and meaningful lyrics. Your lyrics should have an intriguing meaning and message which people can relate too.It should be short. My first request is “ I need a rap song about"+text+".”"
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
                        write_output_to_file(response, filename)
                        save_text_as_audio(response, audio_f)
                    elif text.lower().startswith("tell"):
                        text="I want you to act as a storyteller for kids. You will come up with entertaining stories that are engaging, imaginative and captivating for kids. It can be fairy tales, educational stories or any other type of stories which has the potential to capture their attention and imagination.Since this is for kids, it should be short. My first request is “"+text+".”"
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
                        write_output_to_file(response, filename)
                        save_text_as_audio(response, audio_f)
                    elif text.lower().startswith("pronunce"):
                        text= "I want you to act as an English pronunciation assistant . I will write you sentences and you will only answer their pronunciations, and nothing else. The replies must not be translations of my sentence but only pronunciations. Do not write explanations on replies. My first sentence is “"+text+"”"
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                    elif text.lower().startswith("meaning"):
                        text="I want you to act as a dictinary.Word is“"+text+".”"
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
                    elif text.lower().startswith("goodbye"):
                        speak_text("Goodbye!")
                        break
                    else :
                        response = generate_response(text)
                        print(f"Julia says: {response}")
                        speak_text(response)
        except sr.UnknownValueError:
            print("Speech recognition could not understand audio")
        except sr.RequestError:
            print("Could not request results from speech recognition service")
        except Exception as e:
            print(f"An error occurred: {e}")    

class Redirect():
    
    def __init__(self, widget):
        self.widget = widget

    def write(self, text):
        self.widget.insert('end', text)
        self.widget.see('end')

def load_user_data():
    try:
        with open('user_data.json', 'r', encoding='utf-8') as file:
            user_data = json.load(file)
    except FileNotFoundError:
        user_data = {}
    return user_data

def save_user_data(user_data):
    with open('user_data.json', 'w', encoding='utf-8') as file:
        json.dump(user_data, file)

def validate_login():
    username = entry_username.get()
    password = entry_password.get()

    user_data = load_user_data()

    if username in user_data and user_data[username] == password:
        messagebox.showinfo("Login Successful", "Welcome, admin!" if username == "admin" else f"Welcome, {username}!")
        if username == "admin":
            show_main_page(admin=True)
        else:
            show_main_page()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password.")

def show_main_page(admin=False):
    window.withdraw()
    main_window.deiconify()

    if admin:
        button_check_users.pack(side=tk.LEFT, padx=10)

def logout():
    main_window.withdraw()
    window.deiconify()

def open_julia_page():
    main_window.withdraw()
    julia_window.deiconify()

def open_reminder_page():
    main_window.withdraw()
    reminder_window.deiconify()

def open_user_list():
    main_window.withdraw()
    user_list_window.deiconify()



def back_to_main():
    julia_window.withdraw()
    reminder_window.withdraw()
    user_list_window.withdraw()
    main_window.deiconify()

def validate_signup():
    global entry_signup_username, entry_signup_password
    username = entry_signup_username.get()
    password = entry_signup_password.get()

    if not username or not password:
        messagebox.showerror("Sign Up Failed", "Please enter a username and password.")
        return

    user_data = load_user_data()

    if username in user_data:
        messagebox.showerror("Sign Up Failed", "Username already exists. Please choose a different username.")
    else:
        user_data[username] = password
        save_user_data(user_data)
        messagebox.showinfo("Sign Up Successful", "Account created successfully. You can now log in.")

def show_signup_page():
    global signup_window, entry_signup_username, entry_signup_password
    window.withdraw()
    signup_window = tk.Toplevel(window)
    signup_window.title("Sign Up")
    signup_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

    signup_frame = tk.Frame(signup_window)
    signup_frame.place(relx=0.5, rely=0.5, anchor="center")

    label_signup_username = tk.Label(signup_frame, text="Username:")
    label_signup_username.pack()

    entry_signup_username = tk.Entry(signup_frame)
    entry_signup_username.pack()

    label_signup_password = tk.Label(signup_frame, text="Password:")
    label_signup_password.pack()

    entry_signup_password = tk.Entry(signup_frame, show="*")
    entry_signup_password.pack()

    button_signup_submit = tk.Button(signup_frame, text="Sign Up", command=validate_signup)
    button_signup_submit.pack(pady=10)

    button_signup_back = tk.Button(signup_frame, text="Back", command=back_to_login)
    button_signup_back.pack(pady=10)

def back_to_login():
    signup_window.withdraw()
    window.deiconify()

# Create the main window
window = tk.Tk()
window.title("Login Page")

# Set window dimensions for an average phone screen
window_width = 320
window_height = 480 
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Create the main page window
main_window = tk.Toplevel(window)
main_window.title("Main Page")
main_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
main_window.withdraw()

# Create the Julia window
julia_window = tk.Toplevel(window)
julia_window.title("Julia Page")
julia_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
julia_window.withdraw()

# Create the Reminder window
reminder_window = tk.Toplevel(window)
reminder_window.title("Reminder Page")
reminder_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
reminder_window.withdraw()

# Create the User List window
user_list_window = tk.Toplevel(window)
user_list_window.title("User List")
user_list_window.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")
user_list_window.withdraw()

# Create and position the widgets for the login page
login_frame = tk.Frame(window)
login_frame.place(relx=0.5, rely=0.5, anchor="center")

label_username = tk.Label(login_frame, text="Username:")
label_username.pack()

entry_username = tk.Entry(login_frame)
entry_username.pack()

label_password = tk.Label(login_frame, text="Password:")
label_password.pack()

entry_password = tk.Entry(login_frame, show="*")
entry_password.pack()

button_login = tk.Button(login_frame, text="Login", command=validate_login)
button_login.pack(pady=10)

button_signup = tk.Button(login_frame, text="Sign Up", command=show_signup_page)
button_signup.pack(pady=10)

# Create and position the widgets for the main page
main_frame = tk.Frame(main_window)
main_frame.place(relx=0.5, rely=0.5, anchor="center")

button_julia = tk.Button(main_frame, text="Julia", command=open_julia_page)
button_julia.pack(pady=10)

button_reminder = tk.Button(main_frame, text="Reminder", command=open_reminder_page)
button_reminder.pack(pady=10)

button_story = tk.Button(main_frame, text="StoryTeller", command=open_julia_page)
button_story.pack(pady=10)

button_pronoun = tk.Button(main_frame, text="Pronouncer", command=open_julia_page)
button_pronoun.pack(pady=10)

button_rap = tk.Button(main_frame, text="Rapper", command=open_julia_page)
button_rap.pack(pady=10)

button_Dict = tk.Button(main_frame, text="Dictionary", command=open_julia_page)
button_Dict.pack(pady=10)

button_logout = tk.Button(main_frame, text="Log Out", command=logout)
button_logout.pack(pady=10)

button_check_users = tk.Button(main_frame, text="Check Users", command=open_user_list)
button_check_users.pack(side=tk.LEFT, padx=10)

# Create and position the widgets for the Julia page
julia_frame = tk.Frame(julia_window)
julia_frame.place(relx=0.5, rely=0.5, anchor="center")

julia_output = tk.Text(julia_frame, height=10, width=30)
julia_output.pack(pady=10)

button_start_julia = tk.Button(julia_frame, text="Start", command=VA)
button_start_julia.pack(pady=10)

button_back_julia = tk.Button(julia_frame, text="Back", command=back_to_main)
button_back_julia.pack(pady=10)

# Create and position the widgets for the Reminder page
reminder_frame = tk.Frame(reminder_window)
reminder_frame.place(relx=0.5, rely=0.5, anchor="center")

reminder_output = tk.Text(reminder_frame, height=10, width=40)
reminder_output.pack(pady=10)



button_start_reminder = tk.Button(reminder_frame, text="Start", command=reminder)
button_start_reminder.pack(pady=10)

button_back_reminder = tk.Button(reminder_frame, text="Back", command=back_to_main)
button_back_reminder.pack(pady=10)

# Create and position the widgets for the User List page
user_list_frame = tk.Frame(user_list_window)
user_list_frame.place(relx=0.5, rely=0.5, anchor="center")

user_list_table = ttk.Treeview(user_list_frame, columns=('Username', 'Password'))
user_list_table.heading('#0', text='ID')
user_list_table.heading('Username', text='Username')
user_list_table.heading('Password', text='Password')
user_list_table.pack(pady=10)

def load_users():
    user_data = load_user_data()
    user_list_table.delete(*user_list_table.get_children())
    for i, (username, password) in enumerate(user_data.items(), start=1):
        user_list_table.insert('', 'end', text=str(i), values=(username, password))

load_users()

button_back_users = tk.Button(user_list_frame, text="Back", command=back_to_main)
button_back_users.pack(pady=10)

# Start the main loop
window.mainloop()
