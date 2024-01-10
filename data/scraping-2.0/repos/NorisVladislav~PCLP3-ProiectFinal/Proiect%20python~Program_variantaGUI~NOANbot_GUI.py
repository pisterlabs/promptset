import tkinter as tk
import openai
 
class NOANbotInterface:
    def __init__(self):
        self.messages = []
        self.window = tk.Tk()
        self.window.title("NOANbot Interface")
 
        self.chat_display = tk.Text(self.window, wrap=tk.WORD, width=50, height=20)
        self.chat_display.pack(padx=10, pady=10)
 
        input_frame = tk.Frame(self.window)
        input_frame.pack(pady=10)
 
        self.input_label = tk.Label(input_frame, text="Message NOANbot: ")
        self.input_label.pack(side=tk.LEFT)
 
        self.input_entry = tk.Text(input_frame, wrap=tk.WORD, width=40, height=1)
        self.input_entry.pack(side=tk.LEFT)
        self.input_entry.insert(tk.END, "Type your question here...")
 
        self.send_button = tk.Button(input_frame, text="→", command=self.send_message, bg='green')
        self.send_button.pack(side=tk.RIGHT)
 
        # Bind the "Enter" key
        self.input_entry.bind("<Return>", self.on_enter_pressed)
 
    def on_enter_pressed(self, event):
        self.send_message()
 
    def send_message(self):
        user_input = self.input_entry.get("1.0", tk.END).strip()
        self.input_entry.delete("1.0", tk.END)
 
        self.messages.append({"role": "user", "content": user_input})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        reply = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": reply})
 
        formatted_user = f"\nUser: {user_input}\n"
        formatted_noanbot = f"\nNOANbot: {reply}\n"
 
        self.display_message(formatted_user)
        self.display_message(formatted_noanbot)
 
    def display_message(self, message):
        self.chat_display.configure(state='normal')
        self.chat_display.insert(tk.END, message)
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)
 
    def run(self):
        self.window.mainloop()
 
if __name__ == "__main__":
    openai.api_key = "YOUR API KEY"
    interface = NOANbotInterface()
    interface.run()
