#!/usr/bin/env python3
import subprocess
from dotenv import load_dotenv
from os import getenv
from sys import platform
import openai
import whisper
import queue
import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog, messagebox, simpledialog
from tkinter import *
from spellchecker import SpellChecker
from functools import partial
from templates import LATEX_1, LATEX_2, HTML_5
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from tkinter import Toplevel, Button


load_dotenv()

openai.api_key = getenv("OPENAI_API_KEY")

q = queue.Queue()

def callback(indata, frames, time, status) :
    q.put(indata.copy())

sample_rate = 44100
stream = sd.InputStream(callback=callback, channels=1, samplerate=sample_rate)
audio_list = []

root = tk.Tk()
root.title("Notes")

textarea = ScrolledText(root)
textarea.pack(fill="both", expand=True)

menu = Menu(root)
file = Menu(menu, tearoff = 0)
comp = Menu(menu, tearoff= 0)
ai = Menu(menu, tearoff=0)
templates = Menu(menu, tearoff=0)
scp = Menu(menu, tearoff=0)
audio_menu = Menu(menu, tearoff=0)

def insert_template(template_name) :
    if template_name == "LaTeX 1":
        textarea.insert("end", LATEX_1)
    elif template_name == "LaTeX 2":
        textarea.insert("end", LATEX_2)
    elif template_name == "HTML":
        textarea.insert("end", HTML_5)

    
def Open():
    root.filename = filedialog.askopenfilename(
        initialdir="./",
        title = "Select file",
        filetypes=(("tex files","*.tex"),("text files", "*.txt"), ("all files", "*.*"))
        )
    file = open(root.filename)
    textarea.insert("end",file.read())
def save():
    pass
def saveAs():
    root.filename = filedialog.asksaveasfile(mode="w", defaultextension=".txt")
    if root.filename is None:
        return
    file_to_save = str(textarea.get(1.0,END))
    root.filename.write(file_to_save)
    root.filename.close()
def exit():
    message = messagebox.askquestion ("Notepad", "Do you want to save?")
    if message == "yes" :
        saveAs()
    else:
        root.destroy()

def pdf_compile ():
    root.filename = filedialog.asksaveasfilename(defaultextension=".tex")
    if root.filename is None:
        return
    tex_file_data = str(textarea.get(1.0,END))
    with open(root.filename, "w") as tex_file:
        tex_file.write(tex_file_data)
    try :
        subprocess.run(["pdflatex",root.filename])
    except subprocess.CalledProcessError as error:
        print(f"Compilation failed with error ", error)
    name = root.filename[:-3]
    if platform.startswith("win") :
        subprocess.run([name+"pdf"])
    else : 
        subprocess.run(["open",name+"pdf"])


def show_dialog(text):
        dialog = tk.Toplevel()
        dialog.geometry("300x200")  
        dialog.title("Spell Check Results")

        scroll_text =ScrolledText(dialog, wrap='word')
        scroll_text.pack(fill='both', expand=True)

        scroll_text.insert('insert', text)

def spell_check():

    spell = SpellChecker()
    text = textarea.get(1.0, END)
    text_no_punctuation = text.replace(",","").replace(".","").replace("!","").replace("?","").replace(":","").replace(";","")
    
    words = text_no_punctuation.split()

    misspelled_words = spell.unknown(words)
    if misspelled_words:
        textarea.tag_config("misspelled", foreground="red", underline=True)

        word_index = 0
        for word in words:
            lower_word = word.lower()
            if lower_word in misspelled_words:
                start_index = text.lower().find(lower_word, word_index)
                end_index = start_index + len(lower_word)
                if start_index >= 0:
                    textarea.tag_add("misspelled", f"1.0+{start_index}c", f"1.0+{end_index}c")
                    word_index = end_index

    if misspelled_words:
        errors = []
        for word in misspelled_words:
            errors.append(f"{word} is misspelled.")
            errors.append(f"Suggestions: {spell.correction(word)}")
            errors.append("\n")

        show_dialog("\n".join(errors))
    else:
        show_dialog("No misspelled words found.")



def text_gen():
    response = simpledialog.askstring(title="Prompt",
                                  prompt="What text would you like to generate?")
    prompts = [
        {"role":"system", "content" : "You are a helpful assistent, your job is to compile notes on a given topic."},
        {"role":"user", "content" : response}
    ] 
    gen_text = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages = prompts,
        max_tokens=8192,
        n=1,
        temperature = 0.5
    )
    generated_text = gen_text.choices[0].message["content"].strip()
    textarea.insert("end", generated_text)


def notes_to_text () :
    notes = textarea.get(1.0,END)
    prompts = [
        {"role":"system", "content" : "You are a helpful assistent, your job is to compile the given notes into a structued text. Please keep in mind to mimic the writing style."},
        {"role":"user", "content": notes}
    ]
    text_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages = prompts,
        max_tokens = 16000 - int(len(notes)/4.75),
        n=1,
        temperature = 0.4
    )

    textarea.insert("end", "\n"+text_response.choices[0].message["content"].strip())

def recording_window ():
    recording_window = Toplevel(root)
    recording_window.geometry("150x100")

    start_button = Button(recording_window, text='Start', command=start_recording)
    start_button.pack()

    stop_button = Button(recording_window, text='Stop', command=lambda: stop_recording(recording_window))
    stop_button.pack()

def start_recording():
    print("Recording started...")
    audio_list.clear()
    stream.start()
    root.after(100, update_audio_list)

def stop_recording(window):
    print("Recording stopped.")
    stream.stop()
    audio = np.concatenate(audio_list, axis=0)
    write('output.wav', sample_rate, audio)
    model = whisper.load_model("base.en")
    result = model.transcribe("output.wav")
    textarea.insert("end", result["text"])
    if platform.startswith("win") :
        subprocess.run(["del","output.wav"])
    else:
        subprocess.run(["rm","output.wav"])
    window.destroy()

def update_audio_list():
    while not q.empty():
        audio_list.append(q.get())
    if stream.active:
        root.after(100, update_audio_list) 

def transcribe_audio () :
    root.filename = filedialog.askopenfilename(
        initialdir="./",
        title = "Select audio file",
        filetypes=(("audio files","*.mp3 *.wav"), ("all files", "*.*"))
        )
    input_file = root.filename
    model = whisper.load_model("base.en")
    result = model.transcribe(input_file)
    textarea.insert("end", result["text"])


file.add_command(label = "Open", command = Open)
file.add_command(label = "Save", command = save)
file.add_command(label = "Save As", command = saveAs)
file.add_separator()
file.add_command(label = "Exit", command = exit)
menu.add_cascade(label = "File", menu = file)

comp.add_command(label = "As PDF",command = pdf_compile)
menu.add_cascade(label = "Compile",menu = comp)

ai.add_command(label="Generate notes", command = text_gen)
ai.add_command(label="Notes into text", command = notes_to_text)
menu.add_cascade(label = "AI", menu = ai)

scp.add_command(label = "Spellcheck",command = spell_check)
menu.add_cascade(label="Spellcheck", menu=scp)

audio_menu.add_command(label="Speech to text", command = recording_window)
audio_menu.add_command(label="Transcribe file", command = transcribe_audio)
menu.add_cascade(label="Audio to text", menu=audio_menu)

templates.add_command(label = "LaTeX 1", command = partial(insert_template, "LaTeX 1"))
templates.add_command(label = "LaTeX 2", command = partial(insert_template, "LaTeX 2"))
templates.add_separator()
templates.add_command(label = "HTML", command = partial(insert_template, "HTML"))
menu.add_cascade(label = "Templates", menu = templates)


root.config(menu = menu)

root.mainloop()

