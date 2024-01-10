import tkinter as tk
from tkinter import scrolledtext
import openai
from dotenv import load_dotenv
import os

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is available
if not api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the .env file.")

openai.api_key = api_key

def get_chat_response(input_text):
    # Get a chat response using OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=input_text,
        max_tokens=150
    )
    return response['choices'][0]['text']

def send_message():
    # Get user input and display it in the chat history
    user_input = entry.get()
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, f"You: {user_input}\n", "user")
    chat_history.update_idletasks()

    # Get a response based on user input
    response = get_chat_response(user_input)

    # Display the response in the chat history with the name "AIGrengo"
    chat_history.insert(tk.END, f"AIGrengo: {response}\n", "bot")
    chat_history.config(state=tk.DISABLED)  # Set the state back to DISABLED
    chat_history.yview(tk.END)  # Scroll to the bottom to show the latest message
    entry.delete(0, tk.END)

# Create a Tkinter window
window = tk.Tk()
window.title("aigrengo")

# Chat history display area
chat_history = scrolledtext.ScrolledText(window, width=50, height=20, state=tk.DISABLED)
chat_history.tag_configure("user", foreground="blue")
chat_history.tag_configure("bot", foreground="red")
chat_history.pack()

# Text box for user input
entry = tk.Entry(window, width=50)
entry.pack()

# Send button
send_button = tk.Button(window, text="Send", command=send_message)
send_button.pack()

# Start the application
window.mainloop()
