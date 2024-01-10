import openai
import json
import tkinter as tk
from tkinter import scrolledtext

def analyze_conversation(data, max_tokens=2048):
    character_info = ""
    for conversation_id, conversation in enumerate(data):
        messages = conversation['conversations']
        for message in messages:
            if message['from'] == 'user':
                message_text = message['value']
                conversation_parts = message_text.split()
                if len(conversation_parts) > max_tokens:
                    chunks = [conversation_parts[i:i + max_tokens] for i in range(0, len(conversation_parts), max_tokens)]
                else:
                    chunks = [conversation_parts]
                for chunk in chunks:
                    message_chunk = " ".join(chunk)
                    if len(message_chunk.split()) > max_tokens:
                        message_chunk = " ".join(message_chunk.split()[-max_tokens:])
                    analysis = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"Here is the current character summary: {character_info}. Please update it with information from this message: {message_chunk}"}
                        ]
                    )
                    character_info = analysis['choices'][0]['message']['content'].strip()
                    text_box.insert(tk.END, character_info + "\n")
    with open('character_summary.txt', 'w') as f:
        f.write(character_info)

def start_analysis():
    openai.api_key = api_key_entry.get()

    with open('stage2process.json') as f:
        data = json.load(f)
    analyze_conversation(data)

root = tk.Tk()

api_key_label = tk.Label(root, text="OpenAI API Key:")
api_key_label.pack()

api_key_entry = tk.Entry(root)
api_key_entry.pack()

start_button = tk.Button(root, text="Start Analysis", command=start_analysis)
start_button.pack()

text_box = scrolledtext.ScrolledText(root)
text_box.pack()

root.mainloop()

