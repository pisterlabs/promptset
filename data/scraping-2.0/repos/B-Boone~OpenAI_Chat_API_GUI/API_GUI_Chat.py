#!/usr/bin/env python3

import openai
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, Text, Scrollbar
import threading
import json
import speech_recognition as sr
from gtts import gTTS
import os
import pygame

# Function to save configuration to a JSON file
def save_configuration(settings, filename='config.json'):
    with open(filename, 'w') as f:
        json.dump(settings, f)

# Function to load configuration from a JSON file
def load_configuration(filename='config.json'):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Initialize Pygame Mixer
pygame.mixer.init()

class CustomDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("OpenAI chat API Configuration")
        self.result = None
        
        # Attempt to load configuration
        loaded_config = load_configuration()

        tk.Label(self, text="API Key:").grid(row=0, column=0)
        self.api_key_entry = tk.Entry(self, width=40)
        self.api_key_entry.grid(row=0, column=1)
        if loaded_config:
            self.api_key_entry.insert(0, loaded_config.get('api_key', ''))  # Populate if config loaded

        tk.Label(self, text="Model:").grid(row=1, column=0)
        self.model_entry = tk.Entry(self, width=40)
        self.model_entry.grid(row=1, column=1)
        if loaded_config:
            self.model_entry.insert(0, loaded_config.get('model', ''))

        tk.Label(self, text="Assistant's Role:").grid(row=2, column=0)
        self.role_entry = tk.Entry(self, width=40)
        self.role_entry.grid(row=2, column=1)
        if loaded_config:
            self.role_entry.insert(0, loaded_config.get('role', ''))

        tk.Label(self, text="Assistant's Name:").grid(row=3, column=0)
        self.name_entry = tk.Entry(self, width=40)
        self.name_entry.grid(row=3, column=1)
        if loaded_config:
            self.name_entry.insert(0, loaded_config.get('name', ''))

        self.ok_button = ttk.Button(self, text="OK", command=self.on_ok)
        self.ok_button.grid(row=4, column=0)
        self.cancel_button = ttk.Button(self, text="Cancel", command=self.destroy)
        self.cancel_button.grid(row=4, column=1)

        # Save Configuration Button
        self.save_config_button = ttk.Button(self, text="Save Configuration", command=self.save_config)
        self.save_config_button.grid(row=5, column=0, columnspan=2)

    def on_ok(self):
        api_key = self.api_key_entry.get().strip()
        model = self.model_entry.get().strip()
        role = self.role_entry.get().strip()
        name = self.name_entry.get().strip()
        self.result = (api_key, model, role, name)
        self.destroy()

    def save_config(self):
        settings = {
            'api_key': self.api_key_entry.get(),
            'model': self.model_entry.get(),
            'role': self.role_entry.get(),
            'name': self.name_entry.get()
        }
        save_configuration(settings)

def request_all_settings(parent):
    dialog = CustomDialog(parent)
    parent.wait_window(dialog)  # Wait for dialog to close
    return dialog.result

# Ensure audio directory exists
audio_dir = os.path.join(os.environ['APPDATA'], 'API_GUI_Chat')
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)
    
# Global variable to control the stopping of speech
stop_speech = False

def speak(text):
    global stop_speech, stop_speech_button
    try:
        is_speaking = True
        # Change Stop Speech button color to yellow (loading)
        stop_speech_button.config(style='Yellow.TButton')
        tts = gTTS(text=text, lang='en', tld='us')
        file_path = os.path.join(audio_dir, 'response.mp3')
        tts.save(file_path)
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        # Change Stop Speech button color to green (speaking)
        stop_speech_button.config(style='Green.TButton')
        while pygame.mixer.music.get_busy():
            if stop_speech:
                pygame.mixer.music.stop()
                break
            pygame.time.Clock().tick(10)
        # Reset Stop Speech button color (not speaking)
        stop_speech_button.config(style='TButton')
        pygame.mixer.quit()
    except Exception as e:
        print(f"ERROR- Exception in speak function: {e}")
        # Reset Stop Speech button color (error occurred)
        stop_speech_button.config(style='TButton')
    finally:
        stop_speech = False
        if os.path.exists(file_path):
            os.remove(file_path)

def stop_speak():
    global stop_speech, stop_speech_button
    stop_speech = True

    # Check if the mixer is initialized before trying to stop the music
    if pygame.mixer.get_init() is not None:
        pygame.mixer.music.stop()
        # Reset Stop Speech button color (stopped)
        stop_speech_button.config(style='TButton')
    else:
        # Reset Stop Speech button color (pygame not initialized)
        stop_speech_button.config(style='TButton')

# Speech Recognition Function
recognizer = sr.Recognizer()
def recognize_speech():
    with sr.Microphone() as source:
        try:
            audio_data = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            messagebox.showinfo("Speech Recognition", "Could not understand audio")
        except sr.RequestError:
            messagebox.showinfo("Speech Recognition", "Could not request results; check your internet connection")
        except sr.WaitTimeoutError:
            messagebox.showinfo("Speech Recognition", "No speech detected within the time limit")
        return ""

def send_voice_message():
    text = recognize_speech()
    if text:
        message_entry.delete("1.0", tk.END)
        message_entry.insert("1.0", text)
        send_message()
        
# Functions to save and load chat context
def save_chat_context(chat_context):
    context_file = os.path.join(audio_dir, 'context.json')
    with open(context_file, 'w') as f:
        json.dump(chat_context, f)

def load_chat_context():
    context_file = os.path.join(audio_dir, 'context.json')
    try:
        with open(context_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

settings_entered = False
api_key = None
model = None
assistant_role = None
assistant_name = None
message_entry = None
messages = None
chat_context = []  # Chat context list

def auto_scroll_text_widget(event, text_widget):
    text_widget.see(tk.END)

def send_message_original():
    global settings_entered, message_entry, messages, chat_context
    if not settings_entered:
        messagebox.showinfo("Settings Required", "Please enter the required settings before using the program.")
        return

    msg = message_entry.get("1.0", tk.END).strip()
    if not msg:
        return

    messages.insert(tk.END, 'You: ' + msg + '\n\n-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-\n\n')
    message_entry.delete("1.0", tk.END)
    
    # Auto-scroll message frame to the bottom
    messages.see(tk.END)

    # Update chat context
    chat_context.append({"role": "user", "content": msg})

    try:
        system_message = "{} Your name is {}.".format(assistant_role, assistant_name)
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": msg},
            ] + chat_context
        )

        reply = response.choices[0].message['content']
        messages.insert(tk.END, f'{assistant_name}: ' + reply + '\n\n-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-\n\n')
        
        # Auto-scroll message frame to the bottom
        messages.see(tk.END)

        # Update chat context
        chat_context.append({"role": "assistant", "content": reply})
        speak(reply)

        # Limit the chat context size
        if len(chat_context) > 10:
            chat_context = chat_context[-10:]

        save_chat_context(chat_context)

    except openai.error.AuthenticationError:
        messages.insert(tk.END, 'Authentication failed: Check your API key.\n\n')
    except openai.error.RateLimitError:
        messages.insert(tk.END, 'Rate limit exceeded: Try again later.\n\n')
    except openai.error.OpenAIError as e:
        messages.insert(tk.END, f'OpenAI Error: {e}\n\n')
    except Exception as e:
        messages.insert(tk.END, f'An unexpected error occurred: {e}\n\n')

def send_message(event=None):
    threading.Thread(target=send_message_original, daemon=True).start()

def clear_chat_history():
    global messages, chat_context
    messages.delete('1.0', tk.END)
    chat_context = []

def create_main_window(settings):
    global settings_entered, api_key, model, assistant_role, assistant_name, message_entry, messages, chat_context, stop_speech, stop_speech_button

    api_key, model, assistant_role, assistant_name = settings[0], settings[1], settings[2], settings[3]
    openai.api_key = api_key  # Set the API key for OpenAI
    settings_entered = True

    root = tk.Tk()
    root.title(f'OpenAI Chat API with {assistant_name}')  # Set the window title

    style = ttk.Style(root)
    style.theme_use('default')
    style.configure('TButton', font=('Helvetica', 12), borderwidth=1)
    style.configure('TEntry', font=('Helvetica', 12), borderwidth=1)
    style.configure('Yellow.TButton', background='yellow', font=('Helvetica', 12), borderwidth=1)
    style.configure('Green.TButton', background='green', font=('Helvetica', 12), borderwidth=1)

    
    messages_frame = tk.Frame(root)
    messages = Text(messages_frame)
    scrollbar = Scrollbar(messages_frame, command=messages.yview)
    messages.config(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    messages.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    messages_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    footer_frame = tk.Frame(root)

    send_button = ttk.Button(footer_frame, text='Send', command=send_message, style='TButton')
    message_entry = Text(footer_frame, height=4, font=('Helvetica', 12))
    message_entry.bind('<KeyRelease>', lambda event: auto_scroll_text_widget(event, message_entry))

    messages.configure(wrap='word')
    message_entry.configure(wrap='word')

    def insert_newline(event=None):
        message_entry.insert(tk.INSERT, '\n')
        return 'break'

    message_entry.bind('<Shift-Return>', insert_newline)
    message_entry.bind('<Return>', send_message)
    message_entry.bind('<KP_Enter>', send_message)
    message_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    send_button.pack(side=tk.RIGHT)
    
    # Voice Input Button
    voice_button = ttk.Button(footer_frame, text='Voice Input', command=send_voice_message)
    voice_button.pack(side=tk.LEFT)
    
    # Stop Speech Button
    stop_speech_button = ttk.Button(footer_frame, text='Stop Speech', command=stop_speak)
    stop_speech_button.pack(side=tk.LEFT)

    # Clear Chat History Button
    clear_button = ttk.Button(footer_frame, text='Clear', command=clear_chat_history, style='TButton')
    clear_button.pack(side=tk.RIGHT)

    footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    # Load and display the chat context
    chat_context = load_chat_context()
    if chat_context:
        for context_message in chat_context:
            role = context_message["role"]
            content = context_message["content"]
            messages.insert(tk.END, f'{role.title()}: {content}\n\n')
            
        # Auto-scroll message frame to the bottom after loading initial content
        messages.see(tk.END)

    root.mainloop()

def show_settings_then_main():
    settings_root = tk.Tk()  # Temporary root for settings dialog
    settings_root.withdraw()  # Hide the root window

    loaded_config = load_configuration()
    if loaded_config:
        settings = (loaded_config['api_key'], loaded_config['model'], loaded_config['role'], loaded_config['name'])
    else:
        settings = request_all_settings(settings_root)

    settings_root.destroy()

    if settings:
        create_main_window(settings)
    else:
        messagebox.showinfo("Settings Not Entered", "Program will exit as settings were not entered.")
        exit()

if __name__ == '__main__':
    show_settings_then_main()