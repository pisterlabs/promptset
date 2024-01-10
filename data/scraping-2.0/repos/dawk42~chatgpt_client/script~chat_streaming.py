import openai
import tkinter as tk
import threading

# Replace 'your-model-id' with your actual GPT-3.5 streaming model ID
model_id = 'your-model-id'
mtvar = 50  # Set your preferred max tokens value
tempvar = 0.7  # Set your preferred temperature value

# Initialize the conversation with a system message
conversation_prompt = "You are a brilliant assistant who is an expert in cybersecurity."
conversation = [{"role": "system", "content": conversation_prompt}]

# Initialize the AI response variable
ai_response = ""

# Function to get AI responses
def get_ai_response():
    global conversation, ai_response
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation,
        max_tokens=mtvar,
        temperature=tempvar,
        top_p=1.0,
        n=1,
        stop=None,
        stream=True  # Enable streaming
    )
    for message in response:
        if message['role'] == 'assistant':
            ai_response = message['content']
            display_text.insert(tk.END, "AI: " + ai_response + "\n")
            display_text.insert(tk.END, "====================" + "\n")
            scroll_to_bottom()

# Function to handle user input when the button is clicked
def on_button_click():
    user_input = input_text.get("1.0", tk.END).strip()
    if user_input != "":
        conversation.append({"role": "user", "content": user_input})
        input_text.delete("1.0", tk.END)  # Clear the user input Text widget
        threading.Thread(target=get_ai_response).start()  # Start streaming in a separate thread

# Create the Tkinter window
window = tk.Tk()
window.title("AI Chat")

# Create Text widgets for user input and AI responses
input_text = tk.Text(window, height=4, width=50)
input_text.pack()
input_text.insert(tk.END, "Type your message here.")

display_text = tk.Text(window, height=20, width=50)
display_text.pack()
display_text.config(state=tk.DISABLED)

# Create a button for sending user input
send_button = tk.Button(window, text="Send", command=on_button_click)
send_button.pack()

# Function to scroll the display text to the bottom
def scroll_to_bottom():
    display_text.see(tk.END)

# Start the Tkinter main loop
window.mainloop()