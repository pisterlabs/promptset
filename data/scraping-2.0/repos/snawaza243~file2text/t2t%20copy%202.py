import tkinter as tk
from tkinter import scrolledtext
import openai

def send_query():
    user_input = input_entry.get("1.0", "end-1c")
    if user_input.strip():
        messages.append({"role": "user", "content": user_input})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, api_key=api_key_entry.get()
        )
        reply = chat.choices[0].message.content
        response_text.insert(tk.END, f"User: {user_input}\nChatGPT: {reply}\n\n")
        messages.append({"role": "assistant", "content": reply})
        input_entry.delete("1.0", tk.END)  # Clear the input field

def reset_all():
    input_entry.delete("1.0", tk.END)
    response_text.delete("1.0", tk.END)
    messages.clear()
    messages.append({"role": "system", "content": "You are an intelligent assistant."})

# Create the main window
window = tk.Tk()
window.title("ChatGPT Assistant")
window.geometry("450x550")

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
api_key_label = tk.Label(window, text="Enter API Key:")
api_key_label.pack(pady=5)
api_key_entry = tk.Entry(window, show="*")
api_key_entry.pack(pady=5)

# Button for sending queries
send_button = tk.Button(window, text="Send", command=send_query)
send_button.pack(side=tk.LEFT, padx=5)

# Button for resetting all fields
reset_button = tk.Button(window, text="Reset All", command=reset_all)
reset_button.pack(side=tk.RIGHT, padx=5)

# Run the Tkinter event loop
window.mainloop()
