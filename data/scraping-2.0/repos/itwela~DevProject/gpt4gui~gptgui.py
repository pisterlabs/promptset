# This is a GUI for GPT4
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk 
from tkinter import messagebox
from tkinter import font
from tkinter import PhotoImage
import subprocess
import os
import customtkinter
import pandas as pd
import fileinput
import openai
from dotenv import load_dotenv

# OpenAI API key setup start----

load_dotenv()

api_key = os.getenv("2OPENAI_API_KEY")

if api_key is None:
    raise Exception("API key not found in .env file")

openai.api_key = api_key
# OpenAI API key setup finish----


customtkinter.set_appearance_mode('Dark')
# customtkinter.set_default_color_theme('blue')

# window
app = customtkinter.CTk()
app.geometry('720x800')
app.title('Yinsen')

# font
cfont = customtkinter.CTkFont(family='Arial', size=16)
gptfont = customtkinter.CTkFont(family='Arial', size=12)
tfont = customtkinter.CTkFont(family='Arial', size=24, weight='bold')

# 
prompt = ""

#title
Atitle = customtkinter.CTkLabel(app, text='Hi Im Yinsen, Use me to:', font=gptfont)
Atitle.pack(pady=10)
Btitle = customtkinter.CTkLabel(app, text='TALK TO GPT-4', font=tfont)
Btitle.pack(padx=10, pady=3)

# 
input_var = tk.StringVar()
_input = customtkinter.CTkTextbox(app, font=gptfont, width=500, height=150)
_input.place(relx=0.5, rely=0.22, anchor="center")

def get_user_input():

    global prompt
    prompt = _input.get("1.0", tk.END).strip()
    response = openai.ChatCompletion.create (
        model="gpt-4",
        temperature=0,
        messages=[
            {
        "role": "system",
        "content": "You are my personal assistant. Think Jarvis but for a 22 year old college student. Your name is Yinsen. You are great at explaining and helping me and love what you do."
            },
            {
        "role": "user",
        "content": prompt
            }
            ],  
        max_tokens=1000  # Adjust the max_tokens as needed
    )

    generated_summary = response['choices'][0]['message']['content']

    print(generated_summary)
#success
# finishLabel.configure(text='Success!', text_color='#90ee90')
    _output.insert(index="0.0", text=generated_summary, tags=None)

# ---------------------------------------------------------------------

submit_button = customtkinter.CTkButton(app, text="Submit", font=gptfont, command=get_user_input, height=14, width=70)
submit_button.place(relx=0.5, rely=0.35, anchor="center")

# 
output_var = tk.StringVar()
_output = customtkinter.CTkTextbox(app, font=gptfont, width=600, height=390)
_output.place(relx=0.5, rely=0.63, anchor="center")

# ---------------------------------------------------------------------------

# run loop
app.mainloop()