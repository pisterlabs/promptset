import tkinter as tk
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)

def get_gpt_response():
    user_content = user_input.get()
    
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    ai_response_var.set(response.choices[0].message.content)

# Create the main window
root = tk.Tk()
root.title("GPT-3.5 Turbo Chat Interface")

# Create and place the input label and entry widget
user_input_label = tk.Label(root, text="Ask me any question:")
user_input_label.pack(pady=10)
user_input = tk.Entry(root, width=50)
user_input.pack(pady=10)

# Create a button to trigger GPT-3.5 Turbo response
response_button = tk.Button(root, text="Get Response", command=get_gpt_response)
response_button.pack(pady=10)

# Create a label to display the AI response
ai_response_var = tk.StringVar()
ai_response_label = tk.Label(root, textvariable=ai_response_var)
ai_response_label.pack(pady=10)

# Run the main loop
root.mainloop()
