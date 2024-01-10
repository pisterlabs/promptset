#!/usr/bin/env python3
# Importaciones
import os
import openai
import pprint
from tkinter import *
from tkinter import ttk
from tkinter import messagebox


openai.api_key = os.environ["OPENAI_API_KEY"]

def update_chat(messages, role, content):
    """Añade un mensaje al historial de conversación."""
    messages.append({"role": role, "content": content})
    return messages


def get_chatgpt_response(messages):
    """Obtiene la respuesta del modelo GPT-4"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=150
    )
    return response['choices'][0]['message']['content']


def display_messages(messages, text_widget):
    text_widget.config(state=NORMAL)
    text_widget.delete("1.0", END)
    for msg in messages:
        if msg['role'] == "user":
            text_widget.insert(END, f"Usuario: {msg['content']}\n")
        elif msg['role'] == 'assistant':
            text_widget.insert(END, f"Asistente: {msg['content']}\n")
        else:
            text_widget.insert(END, f"{msg['role']}: {msg['content']}\n")
    text_widget.config(state=DISABLED)

def main():
    root = Tk()
    root.title("Asistente Virtual")
    root.geometry("600x400")

    frame = ttk.Frame(root)
    frame.pack(fill=BOTH, expand=1)

    scrollbar = Scrollbar(frame)
    messagebox_text = Text(frame, wrap=WORD, yscrollcommand=scrollbar.set, state=DISABLED)
    messagebox_text.pack(side=LEFT, fill=BOTH, expand=1)
    scrollbar.pack(side=RIGHT, fill=Y)
    scrollbar.config(command=messagebox_text.yview)

    entry_frame = ttk.Frame(root)
    entry_frame.pack(side=LEFT, fill=X, padx=10, pady=10)

    user_entry = Entry(entry_frame, width=60)
    user_entry.pack(side=LEFT)
    user_entry.focus_set()

    def on_send_click():
        user_text = user_entry.get()
        if user_text.strip() != "":
            update_chat(messages, "user", user_text)
            display_messages(messages, messagebox_text)
            user_entry.delete(0, END)
            messagebox_text.see("end")

            ai_text = get_chatgpt_response(messages)
            update_chat(messages, "assistant", ai_text)
            display_messages(messages, messagebox_text)
            messagebox_text.see("end")

    send_button = ttk.Button(entry_frame, text="Enviar", command=on_send_click)
    send_button.pack(side=LEFT)
    root.bind('<Return>', lambda event: on_send_click())

    intro_message = "¡Hola! Soy tu asistente virtual. ¿En qué puedo ayudarte hoy?"
    update_chat(messages, "assistant", intro_message)
    display_messages(messages, messagebox_text)

    root.mainloop()


if __name__ == "__main__":
    messages = []  # Almacenamiento de mensajes
    main()
