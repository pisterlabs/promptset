import openai, json
import tkinter as tk
from tkinter import filedialog, scrolledtext
import threading

# Configuration
mykeypath='..//..//mykey.json'
MODEL = "gpt-4"

# Load API key
with open(mykeypath, 'r') as f:
    data = json.load(f)
openai.api_key = data['key']

# Load prompts
with open("prompts.json") as fp:
    data = json.load(fp)
    prompts = {prompt['name']: {'system': prompt['system'], 'user': prompt['user']} for prompt in data['prompts']}

def copy_edit(text):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt_text.get("1.0", tk.END)
            },
            {"role": "user", "content": text},
        ],
        max_tokens=int(max_tokens_var.get()),
        n=1,
        stop=None,
        temperature=0,
    )
    return response['choices'][0]['message']['content']

def grammar_rev():
    i=0
    text=text_area1.get('1.0', tk.END).split("\n")
    outuput_text=""
    while i < len(text):
        update_gui("linia:"+str(i))
        edited_text = copy_edit("\n".join(text[i:i+chunk_size_var.get()]))
        outuput_text += edited_text
        i += chunk_size_var.get() 
    text_area2.insert(tk.END, outuput_text)
    update_gui("Procés acabat")

def update_gui(message):
    status_text.insert(tk.END, message + '\n')
    status_text.see(tk.END)

def load_file():
    file_path = filedialog.askopenfilename(filetypes=[('Text Files', '*.txt'), ('Markdown Files','*.md') ])    
    if not file_path:
        return
    text_area1.delete('1.0', tk.END)
    with open(file_path, 'r') as file:
        text_area1.insert(tk.END, file.read())

def save_file():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[('Text Files', '*.txt')])
    if not file_path:
        return
    with open(file_path, 'w') as file:
        file.write(text_area2.get('1.0', tk.END))

def load_prompt():
    selected_prompt = prompt_var.get()
    system_prompt_text.delete('1.0', tk.END)
    system_prompt_text.insert('1.0', prompts[selected_prompt]['system'])
    user_prompt_text.delete('1.0', tk.END)
    user_prompt_text.insert('1.0', prompts[selected_prompt]['user'])

def save_prompt():
    selected_prompt = prompt_var.get()
    prompts[selected_prompt] = {'system': system_prompt_text.get('1.0', tk.END), 'user': user_prompt_text.get('1.0', tk.END)}
    with open("prompts.json", "w") as fp:
        json.dump({'prompts': [{'name': k, 'system': v['system'], 'user': v['user']} for k, v in prompts.items()]}, fp)

def start_process():
    process_thread = threading.Thread(target=grammar_rev)
    process_thread.start()

root = tk.Tk()

# Menu
menu = tk.Menu(root)
root.config(menu=menu)
file_menu = tk.Menu(menu)
menu.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Open", command=load_file)
file_menu.add_command(label="Save", command=save_file)

# GUI components
button_frame = tk.Frame(root)
button_frame.pack(fill=tk.X)

chunk_size_var = tk.IntVar(value=120)  # Define chunk_size_var
chunk_size_label = tk.Label(button_frame, text="Chunk Size")
chunk_size_label.pack(side=tk.LEFT)
chunk_size_entry = tk.Entry(button_frame, textvariable=chunk_size_var)
chunk_size_entry.pack(side=tk.LEFT)

max_tokens_var = tk.StringVar(value='4000')  # Define max_tokens_var
max_tokens_label = tk.Label(button_frame, text="Max Tokens")
max_tokens_label.pack(side=tk.LEFT)
max_tokens_entry = tk.Entry(button_frame, textvariable=max_tokens_var)
max_tokens_entry.pack(side=tk.LEFT)

prompt_var = tk.StringVar()  # Define prompt_var
prompt_menu = tk.OptionMenu(button_frame, prompt_var, *prompts.keys())
prompt_menu.pack(side=tk.LEFT)

system_prompt_label = tk.Label(button_frame, text="System Prompt:")
system_prompt_label.pack(side=tk.LEFT)
system_prompt_text = tk.Text(button_frame, height=2, width=30)
system_prompt_text.pack(side=tk.LEFT)

user_prompt_label = tk.Label(button_frame, text="User Prompt:")
user_prompt_label.pack(side=tk.LEFT)
user_prompt_text = tk.Text(button_frame, height=2, width=30)
user_prompt_text.pack(side=tk.LEFT)

load_prompt_button = tk.Button(button_frame, text="Load Prompt", command=load_prompt)
load_prompt_button.pack(side=tk.LEFT)

save_prompt_button = tk.Button(button_frame, text="Save Prompt", command=save_prompt)
save_prompt_button.pack(side=tk.LEFT)

process_button = tk.Button(button_frame, text='Run process', command=start_process)
process_button.pack(side=tk.LEFT)


text_area_frame = tk.Frame(root)
text_area_frame.pack(fill=tk.BOTH, expand=True)

text_area1 = tk.Text(text_area_frame, wrap='word', width=80, bg='white', fg='black')
text_area1.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)

text_area2 = tk.Text(text_area_frame, wrap='word', width=80, bg='black', fg='white')
text_area2.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


status_text = scrolledtext.ScrolledText(root, wrap='word', bg='lightgray',fg='black')
status_text.pack(expand=True, fill='both')

process_button = tk.Button(root, text='Run process', command=start_process)
process_button.pack()

root.mainloop()
