import threading
import time
import tkinter as tk

from config.openai_key import *
from ai_module import openai_operations

setup_openai_key()


class ChatGUI:
    def __init__(self, main_root, username, bedrijfname):
        self.root = main_root
        self.current_username = username
        self.current_bedrijfname = bedrijfname
        self.root.title("ChatBot")

        # Clear previous UI
        for widget in self.root.winfo_children():
            widget.destroy()

        # Create chat history text box with horizontal scrollbar
        self.chat_history = tk.Text(main_root, state=tk.DISABLED, wrap=tk.WORD,padx=(20),pady=(7))
        self.chat_history.pack(expand=True, fill=tk.BOTH)

        self.chat_history.tag_configure('user', background='white')
        self.chat_history.tag_configure('bot', background='#f7f7f7')

        # Create the horizontal scrollbar
        scrollbar = tk.Scrollbar(main_root, orient=tk.HORIZONTAL, command=self.chat_history.xview)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.chat_history['xscrollcommand'] = scrollbar.set


        # Create the Frame where the user input box and send button are located
        input_frame = tk.Frame(main_root)
        input_frame.pack(fill=tk.BOTH)

        # Create user input box
        self.user_input = tk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.user_input.bind("<Return>", self.send_message)

        # Create a send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def send_message(self, event=None):
        user_message = self.user_input.get()
        self.update_chat_history(f"{self.current_username}: {user_message}", 'user')
        self.user_input.delete(0, tk.END)
        threading.Thread(target=self.get_bot_response, args=(user_message,)).start()

    def get_bot_response(self, user_message):
        bot_response = openai_operations.run_conversation(user_message)
        self.chat_history.after(0, self.update_chat_history, f"{self.current_bedrijfname}_ChatBot: {bot_response}",
                                'bot')

    def update_chat_history(self, message, message_type):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n", message_type)
        self.chat_history.yview(tk.END)  # Auto-scrolls to the end
        self.chat_history.config(state=tk.DISABLED)


# if __name__ == "__main__":
#     # If you want to use UI un comment following 3lines codes!
#     root = tk.Tk()
#     root.configure(bg='white')
#     app = ChatGUI(root)
#     root.mainloop()

    # to use console use following 3 line codes

    # while True:
    #     message = input("Gebruiker:")
    #     user_message_timestamp = time.time()
    #
    #     print("Chatbot" + openai_operations.run_conversation(message))
    #
    #     bot_response_timestamp = time.time()
    #     response_time = bot_response_timestamp - user_message_timestamp
    #     print(f"Response time: {response_time} seconds")
