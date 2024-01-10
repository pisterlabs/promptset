import openai
import tkinter as tk
from tkinter import scrolledtext

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

def send_message(message_log):
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=message_log,
        temperature=0.5,
    )
    for choice in response.choices:
        if "text" in choice:
            return choice.text
    return response.choices[0].message.content

def main():
    message_log=[
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    def on_send():
        user_input=user_entry.get()
        user_entry.delete(0, tk.END)
        
        if user_input.lower() == "quit":
            window.destroy()
            return
    
        message_log.append({"role": "user", "content": user_input})
        response=send_message(message_log)

        message_log.append({"role": "assistant", "content": response})
        conversation.insert(tk.END, f"You: {user_input}\n")
        conversation.insert(tk.END, f"AI assistant: {response}\n")
        conversation.see(tk.END)

    window=tk.Tk()
    window.title("AI Assistant")
    
    conversation=scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20)
    conversation.grid(row=0, column=0, padx=10, pady=10)
    
    user_entry=tk.Entry(window)
    user_entry.grid(row=1, column=0, padx=10, pady=10)
    
    send_button=tk.Button(window, text="Send", command=on_send)
    send_button.grid(row=1, column=1, padx=10, pady=10)
    
    window.bind('<Return>', lambda event: on_send())
    window.mainloop()

if __name__ == "__main__":
    main()