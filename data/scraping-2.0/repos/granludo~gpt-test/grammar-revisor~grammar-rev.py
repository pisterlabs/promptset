import openai, json
import tkinter as tk
from tkinter import filedialog,  scrolledtext
import threading


# gramar-rev.py 
# by Marc Alier 2023
# This script uses the OpenAI API to make a gramatical and stype revision of a text
# it oututs a file with the revision, plus a folder with a visualizations of the revision 
# compared to the original text.
# licensed under the GNU General Public License v3.0
# you need to have a key to use the OpenAI API
# you can get one here: https://beta.openai.com/docs/developer-quickstart/api-key
# this program looks for the key in a file stated in the mykeypath variable 

mykeypath='..//..//mykey.json'
## sample content for the file mykey.json
# {
#  "key": "copy_your_key_here"
# }


MODEL = "gpt-4" # this model is way more expensive


# MODEL = "gpt-3.5-turbo"

def copy_edit(text):
    query=f"{text}"
    print_status("\n\n\n enviant a openai:-----------------------------------------------------------------------------------------------")
    print_status("Query: "+query)

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": prompt_text.get("1.0", tk.END)
            },
            {"role": "user", "content": query},
        ],
        max_tokens=4000,
        n=1,
        stop=None,
        temperature=0,
    )
    print_status("\n\n\nOpenai a respon:-----------------------------------------------------------------------------------------------")
    print_status(response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']

def grammar_rev(lines_per_chunk=120):
    i=0
    text=text_area1.get('1.0', tk.END).split("\n")
    outuput_text=""
    while i < len(text):
        print_status("linia:"+str(i))
        edited_text = copy_edit("\n".join(text[i:i+lines_per_chunk]))
        outuput_text+=edited_text
        i += lines_per_chunk 
    print_status("Procés acabat")
    text_area2.delete('1.0', tk.END)
    text_area2.insert(tk.END, outuput_text)

def print_status(message):
    root.after(0, update_gui, message)
    

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt'), ('Markdown Files','*.md') ])    
    if not file_path:
        return
    text_area1.delete('1.0', tk.END)
    with open(file_path, 'r') as file:
        text_area1.insert(tk.END, file.read())

def get_text():
    return text_area.get('1.0', tk.END)

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[('Text Files', '*.txt')])
    if not file_path:
        return
    with open(file_path, 'w') as file:
        file.write(text_area2.get('1.0', tk.END))


def update_gui(message):
    status_text.insert(tk.END, message + '\n')
    status_text.see(tk.END)

def start_process():
    process_thread = threading.Thread(target=grammar_rev)
    process_thread.start()

root = tk.Tk()
# Create a top frame for the buttons
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X, side=tk.TOP)

# Create buttons
some_button = tk.Button(button_frame, text='Run process', command=lambda: start_process())
some_button.pack(side=tk.LEFT)
load_button = tk.Button(button_frame, text='Load file', command=load_file)
load_button.pack(side=tk.LEFT)
save_button = tk.Button(button_frame, text='Save file', command=save_file)
save_button.pack(side=tk.LEFT)

# Create a frame for text areas
text_area_frame = tk.Frame(root)
text_area_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

# Create first text area with a title and scrollbar
frame1 = tk.Frame(text_area_frame)
frame1.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
text_area1_title = tk.Label(frame1, text="Fitxer Origen")
text_area1_title.pack()
scrollbar1 = tk.Scrollbar(frame1)
text_area1 = tk.Text(frame1, wrap='word', width=80, yscrollcommand=scrollbar1.set, bg='white', fg='black')
scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
text_area1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar1.config(command=text_area1.yview)

# Create second text area with a title and scrollbar
frame2 = tk.Frame(text_area_frame)
frame2.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT)
text_area2_title = tk.Label(frame2, text="Sortida generada")
text_area2_title.pack()
scrollbar2 = tk.Scrollbar(frame2)
text_area2 = tk.Text(frame2, wrap='word', width=80, yscrollcommand=scrollbar2.set, bg='black', fg='white')
scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
text_area2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar2.config(command=text_area2.yview)

# Create a frame for status messages
status_frame = tk.Frame(root)
status_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
status_label = tk.Label(status_frame, text='Status Messages:')
status_label.pack()
status_text = scrolledtext.ScrolledText(status_frame, wrap='word', bg='lightgray',fg='black')
status_text.pack(expand=True, fill='both')

# Create a frame for prompt
prompt_frame = tk.Frame(root)
prompt_frame.pack(fill=tk.BOTH, side=tk.BOTTOM)
prompt_label = tk.Label(prompt_frame, text='Prompt:')
prompt_label.pack()
prompt_text = tk.Text(prompt_frame, height=3, bg='white', fg='black')
prompt_text.pack(fill=tk.X)

with open("defaultprompt.json") as fp:
    data = json.load(fp)
    defaultprompt=data['defaultprompt']
    prompt_text.insert("1.0", defaultprompt)

with open(mykeypath, 'r') as f:
    data = json.load(f)
    print("OK")

# Initialize the OpenAI API
openai.api_key = data['key']

root.mainloop()

# 
