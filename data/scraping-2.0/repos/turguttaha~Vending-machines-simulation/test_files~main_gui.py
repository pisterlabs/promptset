import time
import tkinter as tk
from simulation_lab.vending_machines_simulations import sell_item_from_all_vm
from config.openai_key import *
from ai_module import openai_operations
import asyncio
import threading
import tracemalloc


class MainGUI:
    def __init__(self, main_root):
        self.root = main_root
        self.root.title("ChatBot")

        # Create chat history text box
        self.chat_history = tk.Text(main_root, state=tk.DISABLED)
        self.chat_history.pack()

        # Create the Frame where the user input box and send button are located
        input_frame = tk.Frame(main_root)
        input_frame.pack(fill=tk.BOTH)

        # Create user input box
        self.user_input = tk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create a send button
        self.send_button = tk.Button(input_frame, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def send_message(self):
        user_message = self.user_input.get()
        self.user_input.delete(0, tk.END)
        self.update_chat_history(f"You: {user_message}")

        # Use this code instead of the following
        bot_response = openai_operations.run_conversation(user_message)
        self.update_chat_history(f"Bot: {bot_response}")

        # if ("payment" in user_message.lower()) or ("betaal" in user_message.lower()):
        #     payment_method_str_array, payment_times_int_array = mongodb_data.get_all_payment_and_times()
        #     vending_machines_operations.payment_methode_analysis(payment_method_str_array, payment_times_int_array)
        #
        #
        # else:
        #     response = ai_module.ChatCompletion.create(
        #         model="gpt-3.5-turbo-0613",
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {"role": "user", "content": user_message}
        #         ]
        #     )
        #
        #     bot_response = response.choices[0].message["content"]
        #     self.update_chat_history(f"Bot: {bot_response}")

    def update_chat_history(self, user_message):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.insert(tk.END, user_message + "\n")
        self.chat_history.config(state=tk.DISABLED)

def simulate_vm_operations():

    while True:
        sell_item_from_all_vm()
        # Wait for 5 minutes
        time.sleep(300)


# Enable tracemalloc
tracemalloc.start()

# Create a thread for the background loop
background_thread = threading.Thread(target=simulate_vm_operations)

# Start the thread
background_thread.start()


def chatbot_console():
    while True:
        message = input("Gebruiker:")

        user_message_timestamp = time.time()

        print(openai_operations.run_conversation(message))

        bot_response_timestamp = time.time()
        response_time = bot_response_timestamp - user_message_timestamp
        print(f"Response time: {response_time} seconds")


def chatbot_ui():
    # If you want to use UI un comment following 3lines codes!

    root = tk.Tk()
    root.configure(bg='white')
    app = MainGUI(root)
    root.mainloop()

if __name__ == "__main__":

    setup_openai_key()

    # chatbot_console()

    chatbot_ui()



