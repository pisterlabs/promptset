import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
import openai

import tkinter as tk
from tkinter import scrolledtext, simpledialog  # Added simpledialog
from tkinter import messagebox
import openai



def send_query():
    user_input = input_entry.get("1.0", "end-1c")
    if user_input.strip():
        messages.append({"role": "user", "content": user_input})
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages, api_key=current_api_key
            )
            reply = chat.choices[0].message.content
            response_text.insert(tk.END, f"User: {user_input}\nChatGPT: {reply}\n\n")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            prompt_user_for_api()

        input_entry.delete("1.0", tk.END)  # Clear the input field

def prompt_user_for_api():
    new_api_key = simple_input("Enter a new API Key:")
    if new_api_key:
        global current_api_key
        current_api_key = new_api_key
        messagebox.showinfo("API Key Updated", "API Key updated successfully.")
    else:
        messagebox.showinfo("API Key Error", "API Key is required.")

def simple_input(prompt):
    return simpledialog.askstring("Input", prompt)

def reset_all():
    input_entry.delete("1.0", tk.END)
    response_text.delete("1.0", tk.END)
    messages.clear()
    messages.append({"role": "system", "content": "You are an intelligent assistant."})

def show_about():
    about_text = "ChatGPT Assistant\n\nVersion: 1.0\nAuthor: Your Name"
    messagebox.showinfo("About", about_text)

# Create the main window
window = tk.Tk()
window.title("ChatGPT Assistant")
window.geometry("450x550")

# Initial chat messages with a system message
messages = [{"role": "system", "content": "You are an intelligent assistant."}]

# Initial API Key (empty, to be filled by the user)
current_api_key = "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"

# Hero section
hero_label = tk.Label(window, text="Get Your Query Answer", font=("Helvetica", 16))
hero_label.pack(pady=10)

# Response text field with scrollbar
response_text = scrolledtext.ScrolledText(window, width=50, height=15, wrap=tk.WORD)
response_text.pack(pady=10)

# Input field section
input_entry = tk.Text(window, width=50, height=5, wrap=tk.WORD)
input_entry.pack(pady=10)

# API Key entry
# api_key_label = tk.Label(window, text="Enter API Key:")
# api_key_label.pack(pady=5)
# api_key_entry = tk.Entry(window, show="*")
# api_key_entry.pack(pady=5)

# Button for sending queries
send_button = tk.Button(window, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=5)

# Button for resetting all fields
reset_button = tk.Button(window, text="Reset All", command=reset_all)
reset_button.pack(side=tk.RIGHT, padx=5)

# Window close button
window.protocol("WM_DELETE_WINDOW", window.destroy)

# About button
about_button = tk.Button(window, text="About", command=show_about)
about_button.pack(side=tk.RIGHT, padx=5)

# Run the Tkinter event loop
window.mainloop()
