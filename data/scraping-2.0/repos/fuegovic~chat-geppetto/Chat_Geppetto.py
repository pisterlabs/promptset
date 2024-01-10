import os
import openai
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog

# Get the path of the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "OPENAI_API_KEY")

with open(file_path, 'r') as file:
    api_key = file.read().strip()

openai.api_key = api_key

messages = [
    {"role": "system", "content": ""},
]

def chatbot(input, temperature, max_length):
    if input:
        messages.append({"role": "user", "content": input})
        try:
            if model_var.get() == 1:
                model = "gpt-3.5-turbo"
                chat = openai.ChatCompletion.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_length,
                )
                reply = chat.choices[0].message.content
            else:
                model = "text-davinci-003"
                prompt = "\n".join([f"{msg['content']}" for msg in messages])
                response = openai.Completion.create(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_length,
                    n = 1,
                    stop=None,
                    format="text",
                    model=model
                )
                reply = response.choices[0].text.strip()
            messages.append({"role": "assistant", "content": reply})
            return reply

#Console error handlings
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            status_widget.config(fg="red")
            status_widget.delete(1.0, tk.END)
            status_widget.insert(tk.END, error_msg)
            status_widget.update_idletasks()
            print(error_msg)
            return None
            
def select_model():
    if model_var.get() == 1:
        print("Selected AI model: gpt-3.5-turbo")
        status_widget.config(fg="#4dd0e1")
        status_widget.delete(1.0, tk.END)
        status_widget.insert(tk.END, "Selected AI model: gpt-3.5-turbo")
    else:
        print("Selected AI model: text-davinci-003")
        status_widget.config(fg="#4dd0e1")
        status_widget.delete(1.0, tk.END)
        status_widget.insert(tk.END, "Selected AI model: text-davinci-003")
    
def send_message(event=None):
    input_text = input_box.get('1.0', tk.END)
    if input_text:
        # Show the "Generating response..." message
        chat_log.config(state=tk.NORMAL)
        chat_log.insert(tk.END, f"Human: \n{input_text}\n", "user")
        response_msg = "Generating response..."
        input_box.delete('1.0', tk.END)
        status_widget.config(fg="white")
        status_widget.delete(1.0, tk.END)
        status_widget.insert(tk.END, response_msg)
        status_widget.update_idletasks()
        try:
            output_text = chatbot(input_text, temperature=creativity_slider.get(), max_length=length_slider.get())
            if output_text:
                chat_log.insert(tk.END, f"Geppetto: \n{output_text}\n\n", "bot")
                chat_log.config(state=tk.DISABLED)
                chat_log.see(tk.END)
                status_widget.delete(1.0, tk.END)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            status_widget.config(fg="red")
            status_widget.delete(1.0, tk.END)
            status_widget.insert(tk.END, error_msg)
            status_widget.update_idletasks()
            print(error_msg)

# Clear Everything
def clear_all():
    global messages
    messages = [{"role": "system", "content": ""}]
    chat_log.config(state=tk.NORMAL)
    chat_log.delete("1.0", tk.END)
    status_widget.delete(1.0, tk.END)
    input_box.delete('1.0', tk.END)
    
# Define a function to handle saving the conversation log
def save_log():
    filename = filedialog.asksaveasfilename(defaultextension=".txt")
    if filename:
        with open(filename, "w") as f:
            f.write(chat_log.get("1.0", tk.END))

# Copy function
def copy():
    text.clipboard_clear()
    text.clipboard_append(text.selection_get())

# Paste function
def paste_text():
    if input_box.tag_ranges("sel"):
        # Get the contents of the clipboard
        content = root.clipboard_get()
        # Delete the selected text
        input_box.delete("sel.first", "sel.last")
        # Insert the clipboard contents at the insertion cursor
        input_box.insert("insert", content)
    else:
        # If there is no selected text, simply insert the clipboard contents at the insertion cursor
        content = root.clipboard_get()
        input_box.insert("insert", content)

# Context menu with only copy    
def show_context_menu1(event):
    # create a menu
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Copy", command=copy)

    # display the menu
    menu.post(event.x_root, event.y_root)

# Context menu with copy and paste 
def show_context_menu2(event):
    # create a menu
    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Copy", command=copy)
    menu.add_command(label="Paste", command=paste_text)

    # display the menu
    menu.post(event.x_root, event.y_root)

def insert_newline(event):
    input_box.insert(tk.INSERT, event.char)
 
# Initialize the chatbot GUI
root = tk.Tk()
root.title("Chat Geppetto")
root.geometry("800x1000")
root.resizable(width=False, height=False)
root.iconbitmap(os.path.join(script_dir, "chatbot.ico")) 
# ^^^Replace with your icon file name^^^

# Set the background color to dark grey
root.configure(bg='#333333')

# Create an IntVar to hold the selected model value
model_var = tk.IntVar(value=1)

# Add two radio buttons for selecting the model
model_frame = ttk.Frame(root)
model_frame.grid(row=3, column=4, rowspan=2, padx=10, pady=10, sticky="NSWE")

# Create a new style for the radiobutton and label widgets
style = ttk.Style()
style.configure('my.TLabel', background='#333333', foreground='white', padding=(0,0,0,0), margin=(0,0,0,0))
style.configure('my.TRadiobutton', background='#333333', foreground='white', padding=(0,0,0,0), margin=(0,0,0,0))

# Add a label for the radio buttons
ttk.Label(model_frame, text="Select Model:", style='my.TLabel').grid(row=0, column=0, sticky="NSEW")

# Create two radio buttons for selecting the model
ttk.Radiobutton(
    model_frame, 
    text="gpt-3.5-turbo", 
    variable=model_var, 
    value=1, 
    command=select_model,
    style='my.TRadiobutton'
).grid(row=1, column=0, sticky="NSEW")

ttk.Radiobutton(
    model_frame, 
    text="text-davinci-003", 
    variable=model_var, 
    value=2, 
    command=select_model,
    style='my.TRadiobutton'
).grid(row=2, column=0, sticky="NSEW")

# Create the input box widget
input_box = tk.Text(root, font=("Arial", 14), bg="#C6C6C6", height=1)
input_box.grid(row=6, column=0, padx=10, pady=10, columnspan=4, sticky="NSWE")

# Set the input box to expand up to 3 lines
input_box.config(wrap="word", height=3)

# Create a scrollbar widget and link it to the input box widget
input_scrollbar = tk.Scrollbar(root, command=input_box.yview)
input_scrollbar.grid(row=6, column=3, pady=10, sticky="NSW", rowspan=1)

# Configure the input box widget to use the scrollbar
input_box.configure(yscrollcommand=input_scrollbar.set)

# bind the right-click event to the chat_log widget
input_box.bind("<Button-3>", show_context_menu2)

# Bind the <Return> event to the input box
input_box.bind("<Return>", send_message)

# Bind the <Shift-Return> event to the insert_newline function
input_box.bind("<Shift-Return>", insert_newline)

# Create the send button widget
send_button = tk.Button(root, text="Send", font=("Arial", 14), command=send_message, bg="#222", fg="white")
send_button.grid(row=6, column=4, padx=10, pady=10, sticky="NSEW")

# Create the chat log widget
chat_log = tk.Text(root, wrap=tk.WORD, font=("Arial", 14), state=tk.DISABLED, bg="#C6C6C6", fg="SystemWindowText")
chat_log.grid(row=0, column=0, padx=10, pady=10, columnspan=4, rowspan=5, sticky="NSEW")

# Create a scrollbar widget and link it to the chat log widget
chat_scrollbar = tk.Scrollbar(root, command=chat_log.yview)
chat_scrollbar.grid(row=0, column=3, pady=10, sticky="NSW", rowspan=5)

# Configure the chat log widget to use the scrollbar
chat_log.configure(yscrollcommand=chat_scrollbar.set)

text = tk.Text(root)

# Bind the right-click event to the chat_log widget
chat_log.bind("<Button-3>", show_context_menu1)

# Create the creativity slider widget
creativity_slider = tk.Scale(root, from_=0.1, to=1.0, resolution=0.1, orient=tk.HORIZONTAL,
                             label="Creativity", font=("Arial", 14), bg="#222", fg="white")
creativity_slider.grid(row=7, column=1, padx=10, pady=10, sticky="WE")

# Set the default value for the creativity slider
creativity_slider.set(0.7)

# Create the length slider widget
length_slider = tk.Scale(root, from_=100, to=4000, resolution=10, orient=tk.HORIZONTAL,
                         label="Max Token", font=("Arial", 14), bg="#222", fg="white")
length_slider.grid(row=7, column=0, padx=10, pady=10, columnspan=1, sticky="EW")

# Set the default value for the length slider
length_slider.set(500)

# Create the save button widget
save_button = tk.Button(root, text="Save Chat", font=("Arial", 14), command=save_log, bg="#222", fg="white")
save_button.grid(row=7, column=4, padx=10, pady=10, sticky="NSEW")

# Create the clear button widget
clear_button = tk.Button(root, text="Clear", font=("Arial", 14), command=clear_all, bg="#222", fg="white")
clear_button.grid(row=0, column=4, padx=10, pady=10, sticky="NEW")

# Create the status_widget
status_widget = tk.Text(root, font=("Arial", 14), bg="#222", fg="white", height=3)
status_widget.grid(row=8, column=0, padx=10, pady=10, columnspan=5, sticky="WE")

text = tk.Text(root)

# bind the right-click event to the chat_log widget
status_widget.bind("<Button-3>", show_context_menu1)

# Make the window resizable
root.resizable(True, True)

# Configure the layout to expand the chat log widget
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

# Add a custom tag for the user's message in the chat log
chat_log.tag_configure("user", justify="left", foreground="blue")

# Add a custom tag for the chatbot's message in the chat log
chat_log.tag_configure("bot", justify="left", foreground="green")

root.mainloop()