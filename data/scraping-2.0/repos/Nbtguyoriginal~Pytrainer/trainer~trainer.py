import os
import json
import openai
import logging
from fpdf import FPDF
import random
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, Button, Label, Text, Scrollbar, END, Entry, StringVar, Toplevel, messagebox


MIN_ITEMS = 200  # Set this to the desired number of items

# Set up logging
logging.basicConfig(filename='conversion_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def process_with_chatgpt(txt_content, api_key):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
            model="gpt-3.5-turbo",
            prompt=txt_content,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"Error processing with ChatGPT: {e}")
        messagebox.showerror("Error", f"Error processing with ChatGPT: {e}")
        return None

def content_to_dataset(validated_content, dataset_file):
    with open(dataset_file, 'w') as file:
        for content in validated_content:
            json.dump(content, file)
            file.write('\n')

def train_model(api_key, training_file):
    try:
        openai.api_key = api_key
        response = openai.File.create(file=open(training_file, "rb"), purpose='fine-tune')
        file_id = response.id
        fine_tuning_response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
        job_id = fine_tuning_response.id
        logging.info(f"Fine-tuning job started with ID: {job_id}")
        return f"Fine-tuning job started with ID: {job_id}"
    except Exception as e:
        logging.error(f"Error starting fine-tuning: {e}")
        messagebox.showerror("Error", f"Error starting fine-tuning: {e}")
        return None

def convert_files(folder_path, api_key):
    dataset_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                txt_content = file.read()
                processed_content = process_with_chatgpt(txt_content, api_key)
                if processed_content:
                    dataset_content.append({"messages": [{"role": "user", "content": txt_content}, {"role": "assistant", "content": processed_content}]})
                    logging.info(f"Processed {filename} with content: {processed_content}")
    content_to_dataset(dataset_content, os.path.join(folder_path, "dataset.json"))
    populate_treeview(tree, folder_path)

def view_logs():
    with open('conversion_log.txt', 'r') as log_file:
        logs = log_file.read()
    log_window = Toplevel(root)
    log_window.title("Conversion Logs")
    log_text = Text(log_window, height=20, width=80)
    log_text.pack(pady=10)
    log_text.insert(END, logs)

def view_edit_txt(folder_path):
    def save_changes(file_path, text_widget):
        with open(file_path, 'w') as file:
            file.write(text_widget.get("1.0", END))

    file_path = filedialog.askopenfilename(initialdir=folder_path, title="Select a TXT file", filetypes=(("TXT files", "*.txt"), ("All files", "*.*")))
    if file_path:
        with open(file_path, 'r') as file:
            content = file.read()
        edit_window = Toplevel(root)
        edit_window.title(f"Editing {os.path.basename(file_path)}")
        edit_text = Text(edit_window, height=20, width=80)
        edit_text.pack(pady=10)
        edit_text.insert(END, content)
        save_button = Button(edit_window, text="Save Changes", command=lambda: save_changes(file_path, edit_text))
        save_button.pack(pady=10)

def convert_files():
    folder_path = filedialog.askdirectory()
    if not folder_path:
        return
    dataset_content = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r') as file:
                txt_content = file.read()
                processed_content = process_with_chatgpt(txt_content, api_key_var.get())
                dataset_content.append(json.dumps({"messages": [{"role": "user", "content": txt_content}, {"role": "assistant", "content": processed_content}]}))
                logging.info(f"Processed {filename} with content: {processed_content}")

def populate_treeview(tree, folder_path):
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            tree.insert("", "end", item, text=item)
            populate_treeview(tree, item_path)
        else:
            tree.insert("", "end", text=item)

root = tk.Tk()
root.title("Model Builder Interface, SUDOBRAIN Training Toolkit UI")
root.geometry('1480x900')
root.configure(bg='#f0f0f0')

canvas = tk.Canvas(root, bg='#f0f0f0', bd=0, highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

def on_resize(event):
    canvas.config(width=event.width, height=event.height)
root.bind("<Configure>", on_resize)

items = []

for _ in range(5):
    x, y = random.randint(0, 600), random.randint(0, 500)
    circle = canvas.create_oval(x, y, x+10, y+10, fill='lightblue', outline='lightblue')
    items.append((circle, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

    rect = canvas.create_rectangle(x, y, x+10, y+10, fill='lightgreen', outline='lightgreen')
    items.append((rect, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

    triangle = canvas.create_polygon(x, y, x+10, y, x+5, y-10, fill='lightyellow', outline='lightyellow')
    items.append((triangle, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

for _ in range(5):
    x, y = random.randint(0, 600), random.randint(0, 500)
    letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    letter_item = canvas.create_text(x, y, text=letter, fill='lightpink')
    items.append((letter_item, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

for _ in range(5):
    x, y = random.randint(0, 600), random.randint(0, 500)
    number = random.choice('0123456789')
    number_item = canvas.create_text(x, y, text=number, fill='lightcoral')
    items.append((number_item, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

symbols = ['!', '@', '#', '$', '%', '^', '&', '*']
for _ in range(5):
    x, y = random.randint(0, 600), random.randint(0, 500)
    symbol = random.choice(symbols)
    symbol_item = canvas.create_text(x, y, text=symbol, fill='lightsalmon')
    items.append((symbol_item, random.choice([-1, 1, 2]), random.choice([-1, 1, 2])))

def create_random_item(x, y):
    item_type = random.choice(['shape', 'letter', 'number', 'symbol'])
    if item_type == 'shape':
        shape = random.choice(['circle', 'rect', 'triangle'])
        if shape == 'circle':
            item = canvas.create_oval(x, y, x+10, y+10, fill='lightblue', outline='lightblue')
        elif shape == 'rect':
            item = canvas.create_rectangle(x, y, x+10, y+10, fill='lightgreen', outline='lightgreen')
        else:
            item = canvas.create_polygon(x, y, x+10, y, x+5, y-10, fill='lightyellow', outline='lightyellow')
    elif item_type == 'letter':
        letter = random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        item = canvas.create_text(x, y, text=letter, fill='lightpink')
    elif item_type == 'number':
        number = random.choice('0123456789')
        item = canvas.create_text(x, y, text=number, fill='lightcoral')
    else:
        symbols = ['!', '@', '#', '$', '%', '^', '&', '*']
        symbol = random.choice(symbols)
        item = canvas.create_text(x, y, text=symbol, fill='lightsalmon')
    dx = random.choice([-1, 1, 2])
    dy = random.choice([-1, 1, 2])
    return item, dx, dy

def create_random_item_at_random_position():
    x, y = random.randint(0, canvas.winfo_width() - 10), random.randint(0, canvas.winfo_height() - 10)
    return create_random_item(x, y)

def update_items():
    global items
    new_items = []
    for item, dx, dy in items:
        coords = canvas.coords(item)
        if len(coords) == 4:  # Rectangles and Ovals
            x1, y1, x2, y2 = coords
            if x1 <= 0 or x2 >= canvas.winfo_width() or y1 <= 0 or y2 >= canvas.winfo_height():
                canvas.delete(item)
                new_item, new_dx, new_dy = create_random_item_at_random_position()
                new_items.append((new_item, new_dx, new_dy))
            else:
                canvas.move(item, dx, dy)
                new_items.append((item, dx, dy))
        elif len(coords) == 2:  # Text items
            x, y = coords
            if x <= 10 or x >= canvas.winfo_width() - 10 or y <= 10 or y >= canvas.winfo_height() - 10:
                canvas.delete(item)
                new_item, new_dx, new_dy = create_random_item_at_random_position()
                new_items.append((new_item, new_dx, new_dy))
            else:
                canvas.move(item, dx, dy)
                new_items.append((item, dx, dy))
    items = new_items

    # Ensure the minimum number of items always exist
    while len(items) < MIN_ITEMS:
        new_item, new_dx, new_dy = create_random_item_at_random_position()
        items.append((new_item, new_dx, new_dy))

    root.after(50, update_items)

root.after(100, update_items)  # Delay the initial call by 100ms


frame = ttk.Frame(canvas)
frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

# API Key Section
api_section_label = ttk.Label(frame, text="API Configuration", font=("Arial", 12, "bold"))
api_section_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

api_label = ttk.Label(frame, text="Enter OpenAI API Key:")
api_label.grid(row=1, column=0, sticky=tk.W, pady=5)

api_key_var = tk.StringVar()
api_entry = ttk.Entry(frame, textvariable=api_key_var, width=50, show='*')
api_entry.grid(row=1, column=1, pady=5, sticky=tk.W+tk.E)

# File and Training Section
file_section_label = ttk.Label(frame, text="File and Training", font=("Arial", 12, "bold"))
file_section_label.grid(row=2, column=0, columnspan=2, pady=(10, 10))

select_button = ttk.Button(frame, text="Select Folder with .txt files", command=convert_files)
select_button.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)

train_button = ttk.Button(frame, text="Start Training", command=lambda: result_text.insert(END, train_model(api_key_var.get(), os.path.join(filedialog.askdirectory(), "dataset.pdf"))))
train_button.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.W+tk.E)

# Logs and Editing Section
logs_section_label = ttk.Label(frame, text="Logs and Editing", font=("Arial", 12, "bold"))
logs_section_label.grid(row=5, column=0, columnspan=2, pady=(10, 10))

view_logs_button = ttk.Button(frame, text="View Conversion Logs", command=view_logs)
view_logs_button.grid(row=6, column=0, pady=5, sticky=tk.W+tk.E)

edit_txt_button = ttk.Button(frame, text="View & Edit TXT Files", command=lambda: view_edit_txt(filedialog.askdirectory()))
edit_txt_button.grid(row=6, column=1, pady=5, sticky=tk.W+tk.E)

# Folder Contents Section
tree_label = ttk.Label(frame, text="Folder Contents:")
tree_label.grid(row=7, column=0, pady=(10, 5), sticky=tk.W)

tree = ttk.Treeview(frame)
tree.grid(row=8, column=0, pady=5, sticky=tk.W+tk.E)

# Results Section
result_label = ttk.Label(frame, text="Results:")
result_label.grid(row=7, column=1, pady=(10, 5), sticky=tk.W)

result_text = tk.Text(frame, height=10, width=50)
result_text.grid(row=8, column=1, pady=5, sticky=tk.W+tk.E)

scrollbar = ttk.Scrollbar(frame, command=result_text.yview, orient=tk.VERTICAL)
scrollbar.grid(row=8, column=2, pady=5, sticky=tk.N+tk.S)

result_text.config(yscrollcommand=scrollbar.set)

root.mainloop()