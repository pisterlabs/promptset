import pyperclip
import tkinter as tk
import functools
import openai
import os
import sys
import json
from threading import Thread
from typing import Generator
from bot import get_explanation, initialize_openai_api_key, update_prompt, add_assistant_message
from tkinter.ttk import Combobox

# 创建窗口时，添加stop_sign属性
def create_window(stop_sign, model_var):
    window = tk.Tk()
    model_var = tk.StringVar(window, value=model_var)  # Initialize StringVar with the value of model_var
    window.title("解释机器")
    ## top most level of the screen
    window.attributes('-topmost', True)
    # Screen size
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    # Min and max size
    min_width = screen_width // 8
    min_height = screen_height // 8
    max_width = screen_width // 2
    max_height = screen_height // 2
    response_text = tk.Text(window, wrap=tk.WORD)
    response_text.pack()
    window.minsize(min_width, min_height)
    window.maxsize(max_width, max_height)
    window.stop_sign = stop_sign 
    window.model_var = model_var  
    window.thread_running = False  # Add this line
    return window, response_text, model_var

def update_window(window, response_text, clipboard_content_prev=''):
    clipboard_content = pyperclip.paste()
    if clipboard_content != clipboard_content_prev:
        start_thread(clipboard_content, response_text, window)
    window.after(500, functools.partial(update_window, window, response_text, clipboard_content))  # Schedule the next update


def update_content(response_text, content):
    response_text.insert(tk.END, content)
    text.see(tk.END)
    
def clear_content(response_text):
    response_text.delete('1.0', tk.END)

def produce_output(text: str, response_text, root, stop_sign: list, model_var):
    model = model_var.get()  # Get the current model
    response = get_explanation(text, model)
    response_history = ""
    for chunk in response:
        if stop_sign[0]:  # Check the stop sign
            break
        if 'choices' in chunk and 'delta' in chunk['choices'][0]:
            delta = chunk['choices'][0]['delta']
            if 'content' in delta:
                response_history += delta['content']
                root.after(0, update_content, response_text, delta['content'])
    add_assistant_message(response_history)
    root.thread_running = False  # Set the flag to False when the thread finishes

def start_thread(clipboard_content, response_text, window):
    if window.thread_running:  # If a thread is already running, return
        return
    clear_content(response_text)
    window.stop_sign[0] = False  # Reset the stop sign
    window.thread_running = True  # Set the flag to True
    Thread(target=produce_output, args=(clipboard_content, response_text, window, window.stop_sign, window.model_var)).start()

def stop_thread(stop_sign: list):
    stop_sign[0] = True

# Check if we're running as a script or frozen executable
if getattr(sys, 'frozen', False):
    application_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
elif __file__:
    application_path = os.path.dirname(__file__)

config_path = os.path.join(application_path, 'config.json')
prompts_path = os.path.join(application_path, 'prompts.json')

def get_config():
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config):
    with open(config_path, 'w') as f:
        json.dump(config, f)

def create_combobox(root, models, model_var):
    combo = Combobox(root, textvariable=model_var, values=models)
    combo.current(models.index(model_var.get()))  # set initial value to the last used model

    def switch_model(event):
        selected_model = combo.get()
        config = get_config()
        config['last_model'] = selected_model
        save_config(config)

    combo.bind("<<ComboboxSelected>>", switch_model)
    combo.pack()

    return combo

def load_prompts():
    with open(prompts_path, 'r',encoding='utf-8') as file:
        data = json.load(file)
        return {prompt['tag']: prompt['text'] for prompt in data['prompts']}

def switch_prompt(listbox, prompts,response_text, stop_sign: list):
    clear_content(response_text)
    stop_sign[0] = True
    selected_prompt = listbox.get(tk.ACTIVE)
    update_prompt(prompts[selected_prompt])

def create_listbox(root, prompts):
    listbox = tk.Listbox(root)
    for prompt in prompts:
        listbox.insert(tk.END, prompt)
    listbox.pack()
    return listbox

def create_switch_button(root, listbox, prompts, response_text, stop_sign: list):
    button = tk.Button(root, text="Switch Prompt", 
                command=lambda: switch_prompt(listbox, prompts, response_text, stop_sign))
    button.pack()
    return button

def main():
    initialize_openai_api_key()
    config = get_config()
    stop_sign = [False]
    window, response_text, model_var = create_window(stop_sign, config['last_model']) 
    model_var = tk.StringVar()
    model_var.set(config['last_model'])  # Set the value of model_var
    update_window(window, response_text)
    # Load models and create the combobox
    models = config['models']
    combo = create_combobox(window, models, model_var)
    # 创建下拉菜单
    prompts = load_prompts()
    listbox = create_listbox(window, prompts)
    button = create_switch_button(window, listbox, prompts, response_text, stop_sign)
    stop_button = tk.Button(window, text="Stop", command=lambda: stop_thread(stop_sign))
    stop_button.pack()
    update_window(window, response_text)
    window.mainloop()

if __name__ == "__main__":
    main()
