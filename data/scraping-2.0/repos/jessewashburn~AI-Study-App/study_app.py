import tkinter as tk
from tkinter import scrolledtext
import pyttsx3
import threading
import openai
import os
import re

# OpenAI API setup
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Global variables
study_guide = ""
current_part_index = 0
parts = []
is_speaking = False
engine = pyttsx3.init()
engine_lock = threading.Lock()

def get_study_guide(notes):
    """
    Retrieve a study guide based on input notes and display it.

    Args:
        notes (str): The input notes provided by the user.
    """
    global study_guide, parts, current_part_index
    try:
        display_area.delete('1.0', tk.END)
        display_area.insert(tk.INSERT, "Processing...")
        root.update_idletasks()  

        full_prompt = "Make a detailed study guide based on the following notes:\n" + notes
        response = openai.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=full_prompt,
            max_tokens=500
        )
        study_guide = response.choices[0].text
        parts = re.split(r'\.\s+', study_guide)
        current_part_index = 0

        display_area.delete('1.0', tk.END)
        display_area.insert(tk.INSERT, study_guide)
    except Exception as e:
        print("An error occurred:", e)
        display_area.insert(tk.INSERT, "\n\nAn error occurred. Please try again.")  



def speak_text():
    """
    Speak the study guide text starting from the current part index.
    """
    global is_speaking, current_part_index, parts
    engine_lock.acquire()
    try:
        for part in parts[current_part_index:]:
            if not is_speaking:
                break
            engine.say(part)
            engine.runAndWait()
            current_part_index += 1
    finally:
        engine_lock.release()

def start_speaking():
    """
    Start speaking the study guide.
    """
    global is_speaking
    if parts and not is_speaking:
        is_speaking = True
        threading.Thread(target=speak_text).start()

def stop_speaking():
    """
    Stop speaking the study guide.
    """
    global is_speaking
    is_speaking = False
    engine.stop() 

def fast_forward():
    """
    Fast forward to the next part of the study guide.
    """
    global current_part_index, parts
    current_part_index = min(len(parts), current_part_index + 1)
    stop_speaking()

def rewind():
    """
    Rewind to the previous part of the study guide.
    """
    global current_part_index
    current_part_index = max(0, current_part_index - 1)
    stop_speaking()

def on_closing():
    """
    Handle the closing of the application window.
    """
    global is_speaking
    is_speaking = False
    root.destroy()

# Initialize TTS engine
engine.setProperty('rate', 150)

# Tkinter GUI setup
root = tk.Tk()
root.title("Study Guide with TTS")
root.state('zoomed')  # Maximized window

# Create frames
top_buffer = tk.Frame(root, height=20)  # Buffer at the top
top_frame = tk.Frame(root)
bottom_frame = tk.Frame(root)
bottom_buffer = tk.Frame(root, height=20)  # Buffer at the bottom
top_buffer.pack(side=tk.TOP, fill=tk.X)
top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)
bottom_buffer.pack(side=tk.BOTTOM, fill=tk.X)

# Top frame layout
left_frame = tk.Frame(top_frame)
right_frame = tk.Frame(top_frame)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Add labels
label_notes = tk.Label(left_frame, text="Input Notes")
label_notes.pack()
label_study_guide = tk.Label(right_frame, text="Study Guide")
label_study_guide.pack()

# Input and Display areas
notes_input = tk.Text(left_frame)
notes_input.pack(fill="both", expand=True)
display_area = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD)
display_area.pack(fill="both", expand=True)

# Buttons in Bottom Frame
button_frame = tk.Frame(bottom_frame)  # Frame to contain buttons for centering
button_frame.pack()
process_button = tk.Button(button_frame, text="Process Notes", command=lambda: get_study_guide(notes_input.get("1.0", tk.END)))
process_button.pack(side=tk.LEFT)
play_button = tk.Button(button_frame, text="Play", command=start_speaking)
play_button.pack(side=tk.LEFT)
stop_button = tk.Button(button_frame, text="Stop", command=stop_speaking)
stop_button.pack(side=tk.LEFT)
ff_button = tk.Button(button_frame, text="Fast Forward 15s", command=fast_forward)
ff_button.pack(side=tk.LEFT)
rw_button = tk.Button(button_frame, text="Rewind 15s", command=rewind)
rw_button.pack(side=tk.LEFT)

# Closing protocol
root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()