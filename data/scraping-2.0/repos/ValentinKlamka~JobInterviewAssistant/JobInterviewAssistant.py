from tkinter import *
import tkinter as tk
import pyaudio
from pydub import AudioSegment
import wave
import threading
from os.path import join, dirname
import os
from openai import OpenAI
import configparser
import keyboard
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use('Agg')


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
WAVE_OUTPUT_FILENAME = "output"
FILEINDEX = 0
memory=[]
frames=[]
recording = False
XLIM = 30000  # Desired xlim
YLIM = 3000  # Desired ylim
data = None



#create items consiting of model name, cost and description
class Model:
    def __init__(self, name,  hint,cost,context_size):
        self.name = name
        self.hint = hint
        self.cost = cost
        self.context_size=context_size


Modellist = []
Modellist.append(Model("gpt-3.5-turbo-1106"," (Recommended)", "$0.002 / 1K tokens","Context-window:16385"))
Modellist.append(Model("gpt-4-1106-preview","", "$0.03 / 1K tokens","Context-window:128000"))

client = OpenAI()

class recordThread (threading.Thread):
   def __init__(self):
      threading.Thread.__init__(self)
   def run(self):
      print ("Starting Record" + self.name)
      record()
      print ("Exiting Record" + self.name)

# thread to transcribe the recorded audio file

class transcribeThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):  
        print ("Starting Transcribe" + self.name)
        transcribe()
        print ("Exiting Transcribe" + self.name)

#thread to get response from gpt-3 with argument transcript

class getResponseThread (threading.Thread):
    def __init__(self,transcript):
        threading.Thread.__init__(self)
        self.transcript = transcript
    def run(self):  
        print ("Starting Response" + self.name)
        getResponse(self.transcript)
        print ("Exiting Response" + self.name)

#on enter start recording
def on_enter(e):
    global button_hovered, left_ctrl_pressed
    if  button_hovered  ^ left_ctrl_pressed: # XOR
        button['background'] = 'green'
        button['text'] = 'Recording'
        button['fg'] = 'black'
        print("* recording")
        start_animation(e)
        global recording
        recording=True
        thread_r = recordThread()
        thread_r.start()

def record():
    global data
    global recording
    while recording:

        data = stream.read(CHUNK)
        frames.append(data)


    return
    
def on_leave(e):
    global button_hovered, left_ctrl_pressed
    if not left_ctrl_pressed and not button_hovered:
        button['background'] = 'red'
        button['text'] = 'Record'
        button['fg'] = 'white'
        global recording
        recording=False
        print("* done recording")
        stop_animation(e)
        global FILEINDEX
        wf = wave.open(WAVE_OUTPUT_FILENAME+str(FILEINDEX)+".wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("wav saved")
        frames.clear()
        thread_t = transcribeThread()
        thread_t.start()

def start_animation(event):
    global ani
    ani = animation.FuncAnimation(fig, update_plot, blit=True)
    canvas_p.draw()

def stop_animation(event):
    global ani
    global accumulated_data
    if ani:
        ani.event_source.stop()
        
        fig.canvas.draw()
        ani = None
        accumulated_data = np.zeros(XLIM, dtype=np.int16)  # Clear accumulated data
        canvas_p.draw()


def check_hover_enter(event):
    global button_hovered
    button_hovered = True
    on_enter(event)

def check_hover_leave(event):
    global button_hovered
    button_hovered = False
    on_leave(event)
   

def check_ctrl_enter(event):
    global left_ctrl_pressed
    left_ctrl_pressed = True
    on_enter(event)

def check_ctrl_leave(event):
    global left_ctrl_pressed
    left_ctrl_pressed = False
    on_leave(event)
   

def transcribe():
    global FILEINDEX
    audio_file = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME+str(FILEINDEX)+".wav")
    #if audio file shorter than 0.2 seconds, delete it and return
    if len(audio_file) < 200:
        try:
            os.remove(WAVE_OUTPUT_FILENAME+str(FILEINDEX)+".wav")
        except:
            print("error while deleting file")
        return
    transcript=""
    for i, chunk in enumerate(audio_file[::400*1000]): #split after 400 seconds
        out_file = f"chunk{FILEINDEX}_{i}.wav"
        chunk.export(out_file, format="wav")
        wavchunk=open(out_file,"rb")
        if len(wavchunk.read()) < 200:
            try:
                os.remove(out_file)
            except:
                print("error while deleting file")
            continue
        #transcribe the audio file
        try:
            part = client.audio.transcriptions.create(
                model="whisper-1", 
                file=wavchunk,
                response_format="text"
            )
        except:
            "Connection error"
            break
        #append the part to the full transcript
        transcript=transcript+part
        #close the audio file
        wavchunk.close()


        #delete the audio file
        try:
            os.remove(out_file)
        except:
            print("error while deleting file")
    

    try:
        os.remove(WAVE_OUTPUT_FILENAME+str(FILEINDEX)+".wav")
    except:
        print("error while deleting file")


    if len(transcript) <= 4:
        print("No transcript found")
        return
    #append transcript to log.txt file
    f = open("log.txt", "a")
    try:
        f.write(transcript)
    except:
        print("error while writing to log.txt")
    f.write("\n")
    f.close()

    #append transcript to text box
    text_box.configure(state='normal')
    text_box.insert(END,transcript)
    text_box.insert(END,"\n")
    text_box.configure(state='disabled')
    #scroll to bottom
    text_box.see(END)
    
    FILEINDEX+=1
    #start response thread
    thread_g = getResponseThread(transcript)
    thread_g.start()
    return

def getResponse(transcript):
    global memory 
    #read config.ini file to get gpt-version
    config = configparser.ConfigParser()
    config.read('config.ini')

    #open and read notes.md file
    try:
        f = open("notes.md", "r")
        notes = f.read()
        f.close()
        if len(notes) > 0:
            notes_promt= {"role": "assistant", "content": "Notes: "+notes}
        else:
            notes_promt= {"role": "assistant", "content": "No Notes Provided"}
    except:
        notes_promt= {"role": "assistant", "content": "No Notes Provided"}


   
    #get gpt-version from config.ini file
    gpt_version = config["GPT-Version"]["gpt-version"]
    try:
        response = client.chat.completions.create(
            model=gpt_version,
            messages=[
                {"role": "system", "content": "Please help to guide me through my job interview. I am the interviewed person. Answer the questions from my perspective. Use my notes, if I provided any. Answer all job interview questions with competence and confidence."},
                notes_promt,
                {"role": "assistant", "content": ''.join(memory)},
                {"role": "user", "content": transcript}
            ],
            stream=True
            )
    except:
        print("Connection error")
        return
    collected_messages = []
    text_box.tag_config('blue', foreground="blue")
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message:
            collected_messages.append(chunk_message)
            text_box.configure(state='normal')
            #insert the chunk_message into the text box in blue color
            text_box.insert(END,chunk_message,'blue')
            text_box.configure(state='disabled')
            update_arrow_visibility()
    text_box.configure(state='normal')
    text_box.insert(END,"\n")
    text_box.configure(state='disabled')
    full_reply_content = ''.join(collected_messages)
    f = open("log.txt", "a")
    try:
        f.write(full_reply_content)
    except:
        print("error while writing to log.txt")
    f.write("\n")
    f.close()
    memory.append(transcript)
    memory.append(full_reply_content)
    return


def on_closing():
    global recording
    recording=False

    stream.stop_stream()
    stream.close()
    p.terminate()
    window.quit()
    window.destroy()

def select(option):
    #change config.ini file to change gpt-version
    config = configparser.ConfigParser()
    config.read('config.ini')
    config['GPT-Version']['gpt-version'] = option
    with open('config.ini', 'w') as configfile:
        config.write(configfile)
    changebold()
    return

def changebold():
    #value of gpt-version in config.ini file
    config = configparser.ConfigParser()
    config.read('config.ini')
    for index in range(file_menu.index("end")):
        if config["GPT-Version"]["gpt-version"] ==Modellist[index].name:

            file_menu.entryconfig(index, font=('TkDefaultFont', 10, 'bold'))
            add_checkmark(file_menu, index)

        else:
            file_menu.entryconfig(index, font=('TkDefaultFont', 10))
            remove_checkmark(file_menu, index)
    return

def add_checkmark(menu, index,checkmark='\u2713'):

    label = menu.entrycget(index, "label")
    if checkmark not in label:
        label += f" {checkmark}"
        menu.entryconfig(index, label=label)

def remove_checkmark(menu, index,checkmark='\u2713'):
    label = menu.entrycget(index, "label")
    if checkmark in label:
        label = label.replace(checkmark, "").strip()
        menu.entryconfig(index, label=label)

    
    return

def send_message(event=None):
    message = entry_box.get("1.0", tk.END).strip()
    if message.strip() != "":
        text_box.config(state=tk.NORMAL)
        text_box.insert(tk.END, f"\n{message}\n")
        f = open("log.txt", "a")
        f.write(message)
        f.write("\n")
        f.close()
        text_box.config(state=tk.DISABLED)
        text_box.see(tk.END)
        entry_box.delete("1.0", tk.END)
        #start response thread
        thread_g = getResponseThread(message+"\n")
        thread_g.start()
    return 


def scroll_to_end(event):
    global at_end
    at_end = True
    text_box.see(tk.END)
    canvas.place_forget()
    canvas.unbind('<Button-1>')
    

def update_arrow_visibility(event=None):
    global at_end
    bottom = float(text_box.index(tk.END))
    visible_bottom = 1.0+float(text_box.index('@0,{} linestart'.format(str(text_box.winfo_height()))))
    if bottom > visible_bottom:
        at_end = False
        canvas.place(in_=text_box, relx=0.5, rely=1.0, anchor=tk.S)
        canvas.bind('<Button-1>', scroll_to_end)

    else:
        at_end = True
        canvas.place_forget()
        canvas.unbind('<Button-1>')


p = pyaudio.PyAudio()


stream = p.open(format = FORMAT,
                channels = CHANNELS,
                rate = RATE,
                input = True,
                input_device_index =2,
                frames_per_buffer = CHUNK
                )

# Initialize figure and plot for the left channel
fig, ax = plt.subplots()
x = np.arange(0, XLIM)
line, = ax.plot(x, np.zeros(XLIM), '-', lw=2)
# do not show axis
ax.axis('off')

ax.set_ylim(-YLIM, YLIM)
ax.set_xlim(0, XLIM)

# Initialize an array to accumulate chunks
accumulated_data = np.zeros(XLIM, dtype=np.int16)

# Animation object reference
ani = None

# Function to update the plot
def update_plot(frame):
    global accumulated_data
    global data
    data = stream.read(CHUNK)
    data_np = np.frombuffer(data, dtype=np.int16)
    
    # Append incoming chunk to accumulated data
    accumulated_data = np.roll(accumulated_data, -CHUNK)
    accumulated_data[-CHUNK:] = data_np[::CHANNELS]  # Extract left channel data
    
    line.set_ydata(accumulated_data)
    
    return line,


window=tk.Tk()

at_end=True

button_hovered = False
left_ctrl_pressed = False

mainframe=tk.Frame(window,bg='white')
mainframe.pack(fill=BOTH, expand=1)
window.title('Job Interview Assistant')  
screen_width = window.winfo_screenwidth()
window.geometry(str(screen_width)+ 'x200')
#it should open on the top of the screen
window.geometry("+0+0")

window.configure(background='white')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

menu_bar = Menu(window)
file_menu = Menu(menu_bar, tearoff=0)
for model in Modellist:
    file_menu.add_command(label=model.name+model.hint + "| "+model.cost+"| " +model.context_size, command=lambda m=model.name: select(m))

file_menu.add_command(label="Quit", command=on_closing) 
menu_bar.add_cascade(label="File", menu=file_menu)
window.config(menu=menu_bar)
changebold()

# Canvas to embed matplotlib plot
#create Childframe on the left side of the window

childframe = tk.Frame(mainframe,bg='white', width=200)
childframe.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
childframe.pack_propagate(False)

canvas_p = FigureCanvasTkAgg(fig, master=childframe)



button = tk.Button(childframe, text='Record', width=20, height=10, bg='red', fg='white')
button.pack(side=BOTTOM)

canvas_p.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


button.bind("<Enter>", check_hover_enter)
button.bind("<Leave>", check_hover_leave)

#when pressing ctrl, trigger check_ctrl_enter, even if out of focus
keyboard.on_press_key("ctrl", check_ctrl_enter)
keyboard.on_release_key("ctrl", check_ctrl_leave)
# to the right side of the button create a scrollable text box
text_box = tk.Text(mainframe, height=10, width=200)

# make it scrollable
scroll = tk.Scrollbar(mainframe, command=text_box.yview)
scroll.pack(side=RIGHT, fill=Y)
text_box.config(yscrollcommand=scroll.set)
#make it non editable
text_box.configure(state='disabled')

# Loading the down arrow image
down_arrow_img = tk.PhotoImage(file='arrow_down.png')  # Replace 'down_arrow.png' with your image file path

# Create a canvas inside the Text widget. change the cursor to finger
canvas = tk.Canvas(text_box, highlightthickness=0, background='white', cursor="left_ptr",width=16, height=16)
canvas_image = canvas.create_image(8, 8, image=down_arrow_img, anchor=tk.CENTER)


text_box.bind('<Configure>', update_arrow_visibility)
text_box.bind('<MouseWheel>', update_arrow_visibility)
text_box.bind('<Button-1>', update_arrow_visibility)
canvas.bind('<Button-1>', update_arrow_visibility)
entry_frame = tk.Frame(mainframe, bg='white')
entry_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, padx=5, pady=5)
text_box.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5, expand=True)

entry_box = tk.Text(entry_frame,height=2,width=200) 
entry_box.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
#make entry box scrollable
scroll = tk.Scrollbar(entry_frame, command=entry_box.yview)
scroll.grid(row=0, column=1, sticky="ns")
entry_box.config(yscrollcommand=scroll.set)

#bind enter key to send message
entry_box.bind("<Return>", send_message)


send_button = tk.Button(entry_frame, text="Send", command=send_message)
send_button.grid(row=0, column=2, padx=5, pady=5, sticky="ew")


entry_frame.columnconfigure(0, weight=1)  # Make entry_box expandable
entry_frame.columnconfigure(1, minsize=50)  # Set a minimum size for the Send button




window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()





