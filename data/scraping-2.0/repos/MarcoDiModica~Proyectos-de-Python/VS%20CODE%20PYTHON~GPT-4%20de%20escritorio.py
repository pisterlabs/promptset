import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import messagebox
import openai
import asyncio
import httpx

openai.api_key = 'API_KEY'

async def gpt4_chat(message):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/engines/davinci-codex/completions",
                headers={"Authorization": f"Bearer {openai.api_key}"},
                json={
                    "prompt": message,
                    "max_tokens": 100,
                    "temperature": 0.5
                }
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            print(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
            print(exc.response.text)
            return "Error: Could not connect to the API."
        except Exception as exc:
            print(f"An error occurred: {exc}")
            return "Error: An unexpected error occurred."
        else:
            data = response.json()
            return data['choices'][0]['text'].strip()

def send_message():
    message = user_entry.get()
    if message != "":
        chat_history.insert(tk.END, "Tú: " + message + '\n\n')
        user_entry.delete(0, tk.END)
        response = asyncio.run(gpt4_chat(message))
        chat_history.insert(tk.END, "GPT-4: " + response + '\n\n')

root = ThemedTk(theme="radiance")

root.title("Chat GPT-4")

# Create a text widget with rounded corners
chat_history = tk.Text(root, bd=1, bg="black", width="50", height="8", font=("Arial", 23), foreground="#00ffff")
chat_history.configure(relief="solid", borderwidth=1, highlightthickness=1, highlightbackground="white")
chat_history.place(x=6,y=6, height=385, width=370)

# Create an entry widget with rounded corners
user_entry = tk.Entry(root, bd=1, bg="black",width="30", font=("Arial", 23), foreground="#00ffff")
user_entry.configure(relief="solid", borderwidth=1, highlightthickness=1, highlightbackground="white")
user_entry.place(x=128, y=400, height=88, width=260)

# Create a send button with rounded corners
send_button = tk.Button(root, text="Enviar", command=send_message, font=("Arial", 12), bg="blue", activebackground="lightblue", bd=0)
send_button.configure(relief="solid", borderwidth=1, highlightthickness=1, highlightbackground="white")
send_button.place(x=6, y=400, height=88)

root.mainloop()