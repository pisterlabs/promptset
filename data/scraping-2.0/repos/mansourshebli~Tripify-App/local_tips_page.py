# @mansourshebli
from tkinter import *
from tkinter import messagebox
import openai
import os
import sys

# Set up OpenAI API key
try:
    openai.api_key = os.environ['OPENAI_API_KEY']
except KeyError:
    # Adding: Instructions on how to obtain an API key
    sys.stderr.write("""
    You haven't set up your API key yet.
    
    If you don't have an API key yet, visit:
    
    https://platform.openai.com/signup

    1. Make an account or sign in
    2. Click "View API Keys" from the top right menu.
    3. Click "Create new secret key"

    Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
    """)
    exit(1)

# Function to clear text fields
def clear_tf():
    # Adding: Clear the place text field and reset continent selection
    place_tf.delete(0, 'end')
    var.set(0)

# Function to generate local tips
def generate_local_tips():
    global place_tf, var

    # Get user inputs
    place = place_tf.get()
    continent = var.get()

    user_message = f"I'm looking for travel tips about {place}. I'm interested in exploring {continent}."

    messages = [
        {"role": "system", "content": "I'm helping you find travel tips for your chosen destination."},
        {"role": "user", "content": user_message}
    ]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    content = response['choices'][0]['message']['content'].strip()

    # Create a result window
    result_window = Toplevel()
    result_window.title(f"Local Tips for {place}")
    result_window.config(bg='#FF0000')

    # Calculate the window size based on content
    window_width = 600
    window_height = 400

    # Get the screen width and height
    screen_width = result_window.winfo_screenwidth()
    screen_height = result_window.winfo_screenheight()

    # Calculate the x and y position to center the window
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Set the window geometry
    result_window.geometry(f'{window_width}x{window_height}+{x}+{y}')

    # Make the result window non-resizable
    result_window.resizable(False, False)

    # Create a scrollable text widget
    text_widget = Text(result_window, wrap=WORD)
    text_widget.pack(fill=BOTH, expand=True)

    lines = content.split('\n')
    for line in lines:
        text_widget.insert(INSERT, line + "\n", 'normal')

    text_widget.tag_configure('normal', font=('Helvetica', 12))

    scroll_bar = Scrollbar(result_window)
    scroll_bar.pack(side=RIGHT, fill=Y)

    text_widget.config(yscrollcommand=scroll_bar.set)
    scroll_bar.config(command=text_widget.yview)

    text_widget.configure(state=DISABLED)

    # Make the result window non-resizable
    result_window.resizable(False, False)

# Create the main window
lt_window = Tk()
lt_window.title('Local Tips')
lt_window.geometry('800x500')
lt_window.config(bg='#00FFFF')

var = IntVar()

# Create the frame for local tips content
lt_frame = Frame(lt_window, padx=50, pady=50, bg='#00FFFF')
lt_frame.grid(row=0, column=0, sticky='nsew')
lt_window.grid_rowconfigure(0, weight=1)
lt_window.grid_columnconfigure(0, weight=1)

title_label = Label(lt_frame, text='Local Tips', font=('Helvetica', 24, 'bold', 'italic'), bg='#00FFFF')
title_label.grid(row=1, column=1, sticky="w")

place_lb = Label(lt_frame, text="Please enter a place:", font=('Arial', 12), bg='#00FFFF')
place_lb.grid(row=3, column=1, pady=40, sticky="w")

place_tf = Entry(lt_frame)
place_tf.grid(row=3, column=2, pady=10, padx=10, sticky="w")

continent_lb = Label(lt_frame, text="Choose place continent:", font=('Arial', 12), bg='#00FFFF')
continent_lb.grid(row=4, column=1, sticky="w")

radio_frame = Frame(lt_frame, bg='#00FFFF')
radio_frame.grid(row=4, column=2, pady=10, padx=10, sticky="w")

asia_continent = Radiobutton(radio_frame, text="Asia", variable=var, value=1, bg='#00FFFF')
africa_continent = Radiobutton(radio_frame, text="Africa", variable=var, value=2, bg='#00FFFF')
europe_continent = Radiobutton(radio_frame, text="Europe", variable=var, value=3, bg='#00FFFF')
other_continents = Radiobutton(radio_frame, text="Other Continent", variable=var, value=4, bg='#00FFFF')

asia_continent.pack(side=LEFT)
africa_continent.pack(side=LEFT)
europe_continent.pack(side=LEFT)
other_continents.pack(side=LEFT)

local_tips_btn = Button(lt_frame, text="Generate Local Tips", command=generate_local_tips, bg="green", fg="white", font=("Arial", 12))
local_tips_btn.grid(row=5, column=1, padx=5, pady=5, sticky="w")

clear_btn = Button(lt_frame, text="Clear", command=clear_tf, bg="blue", fg="white", font=("Arial", 12))
clear_btn.grid(row=5, column=2, padx=5, pady=5, sticky="w")

exit_btn = Button(lt_frame, text='Exit', command=lt_window.destroy, bg="red", fg="white", font=("Arial", 12))
exit_btn.grid(row=5, column=3, padx=5, pady=5, sticky="w")

# Start the main event loop
lt_frame.mainloop()
