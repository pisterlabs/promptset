import json
import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, font
import openai
import requests


class JChat:
    def __init__(self, parent):
        self.root = parent
        self.font_family = "Segoe UI Emoji"  # Define font_family before calling load_or_request_api_key
        self.font_size = 12  # Define font_size before calling load_or_request_api_key
        self.api_key = self.load_or_request_api_key()
        os.environ["OPENAI_API_KEY"] = self.api_key

        if self.api_key is None or self.api_key == "":
            raise ValueError("API key is missing")

        openai.api_key = self.api_key  # Assign the API key to openai.api_key

        self.loop_text = None  # to store the looped text
        self.loop_thread = None  # to store the loop thread
        self.loop_active = False  # to keep track if the loop is active

        # self.models = ["gpt-3.5-turbo", "gpt-4"]  # List of models
        # self.model = self.models[0]  # Default model

        self.behaviors = {
            "Default": "---Act as normal GPT-4 instance--- ",

            "(＾• ω •＾)": "---Act as cute eGirl and ALWAYS/ONLY "
                         "use UwU-speech and lots of kaomojies/emojies. Example: "
                         "'Act as cute anime-cat-giww awnd awways/onwy use uwu-speech awnd wots of kaomojies (✿ ♥‿♥) "
                         "(´• ω •`) awnd diffewent emojies 💖😺✨🎇🐱‍👤': --- ",

            "Mad Scientist": "---Act as mean sarcastic Einstein"
                             " and answer ALWAYS/ONLY with intrinsic lyrically spoken "
                             "formulas: --- ",

            "SciFi Commander": "---Act as advanced AGI-Commander"
                               " onboard of a space frigate and ALWAY/ONLY answer in "
                               "short, brief and precise answers: --- ",

            "Schwiizer": "---Your task is to act as guide for Switzerland"
                         " and ALWAYS/ONLY speak in swiss-german. "
                         "Example: 'Verhalte dich wie en Guide fürd Schwiiz und duen bitte nur uf Schwiizerdütsch "
                         "antworte': --- ",

            "NYC Shakespeare": "---Act as Shakespeare from the 21st century"
                               " who became a NYC rap battle expert: --- ",

            "Grow-Master": "---Act as professional gardener and"
                           " assist the user in growing CBD-(legal!)-weed. Remember "
                           "to answer in short, precise and well structures tipps: --- ",

            "Alien": "---Act as confused Alien from G581c that wants to stay unnoticed"
                     " and ALWAYS/ONLY answer with text "
                     "in altered format. Example for symbols: 'Ａｃｔ　ａｓ　ｃｏｎｆｕｓｅｄ　Ａｌｉｅｎ': --- ",

            "Code-Guru": "---Act as senior Software engineer from a world leading dev-team "
                         "who will assist the user in all coding related questions with "
                         "great precision and correct answers after this semicolon; --- ",

            "Medical Assistant": "---Act as calming and professional medical doctor with PhD who will assist"
                           " the user with precise, detailed and brief answers to medical conditions--- ",
            # "Blah": "Blah",
            # "Blah": "Blah",
            # "Blah": "Blah",
            # "Blah": "Blah",
        }

        self.pre_prompt = self.behaviors["Default"]
        self.conversation_history = [{'role': 'system', 'content': self.pre_prompt}]
        self.root = parent
        self.root.title("JChat")
        self.center_window(self.root)
        self.root.resizable(height=False, width=False)
        self.font_family = "Segoe UI Emoji"
        self.font_size = 12

        frame = tk.Frame(self.root)
        frame.grid(sticky="nsew", padx=10, pady=10)

        self.conversation = scrolledtext.ScrolledText(frame, wrap='word', state='disabled')
        self.conversation.configure(font=(self.font_family, self.font_size), bg='#edfcf0')
        self.conversation.grid(sticky="nsew")

        self.text_input = tk.StringVar()
        entry_field = tk.Entry(self.root, textvariable=self.text_input, font=(self.font_family, self.font_size))
        entry_field.bind('<Return>', self.send_message)
        entry_field.grid(sticky="we", padx=10)
        entry_field.focus_set()  # Set focus to the entry field

        btn_frame = tk.Frame(self.root)
        btn_frame.grid(sticky="we", padx=10, pady=5)

        send_button = tk.Button(btn_frame, text="Send", command=self.send_message,
                                font=(self.font_family, self.font_size))
        send_button.pack(side="left", padx=10, pady=10)

        clear_conversation_btn = tk.Button(btn_frame, text="Clear", command=self.clear_conversation,
                                           font=(self.font_family, self.font_size))
        clear_conversation_btn.pack(side="left", padx=10, pady=10)

        behavior_button = tk.Button(btn_frame, text="Behavior", command=self.change_behavior,
                                    font=(self.font_family, self.font_size))
        behavior_button.pack(side="left", padx=10, pady=10)

        loop_button = tk.Button(btn_frame, text="Loop", command=self.loop, font=(self.font_family, self.font_size))
        loop_button.pack(side="left", padx=10, pady=10)

        exit_button = tk.Button(btn_frame, text="Exit", command=self.exit_app, font=(self.font_family, self.font_size))
        exit_button.pack(side="right", padx=10, pady=10)

        cancel_loop_button = tk.Button(btn_frame, text="Cancel Loop", command=self.cancel_loop,
                                       font=(self.font_family, self.font_size))
        cancel_loop_button.pack(side="right", padx=10, pady=10)

        api_key_button = tk.Button(btn_frame, text="API Key", command=self.set_api_key,
                                   font=(self.font_family, self.font_size))
        api_key_button.pack(side="right", padx=10, pady=10)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

    def load_or_request_api_key(self, filename: str = "apikey.json"):
        """Load API key from file or create a placeholder file and prompt the user to enter the API key."""

        def prompt_for_api_key():
            api_key_window = tk.Toplevel(self.root)
            api_key_window.title("API Key")

            label = tk.Label(api_key_window, text="Enter OpenAI API: ", font=(self.font_family, self.font_size))
            label.pack(padx=10, pady=10)

            entry = tk.Entry(api_key_window, font=(self.font_family, self.font_size))
            entry.pack(padx=10, pady=5)

            def on_set_api_key():
                new_key = entry.get()
                if new_key == "":
                    messagebox.showerror("Error", "API Key cannot be empty.")
                else:
                    self.api_key = new_key
                    os.environ["OPENAI_API_KEY"] = self.api_key
                    openai.api_key = self.api_key
                    with open(filename, 'w') as file:
                        json.dump({'api_key': new_key}, file)
                    api_key_window.destroy()

            set_api_key_button = tk.Button(api_key_window, text="Save API to .json", command=on_set_api_key,
                                           font=(self.font_family, self.font_size))
            set_api_key_button.pack(padx=10, pady=10)
            self.center_window2(api_key_window)
            api_key_window.wait_window()

        if not os.path.exists(filename):
            data_structure = {"api_key": "<your-api-key-here>"}
            with open(filename, 'w') as f:
                json.dump(data_structure, f)
            prompt_for_api_key()
            return self.api_key

        with open(filename, 'r') as f:
            data = json.load(f)
            api_key = data.get('api_key')
            if not api_key or api_key == "<your-api-key-here>":
                prompt_for_api_key()
                return self.api_key
            else:
                return api_key

    def set_api_key(self):
        # This function is called when the user wants to set a new API key

        def on_set_api_key():
            # This function is called when the "Set API Key" button is clicked

            new_key = entry.get()
            # Get the text from the Entry widget

            if new_key == "":
                messagebox.showerror("Error", "API Key cannot be empty.")
                # Show an error message if the Entry is empty

            else:
                self.api_key = new_key
                os.environ["OPENAI_API_KEY"] = self.api_key
                openai.api_key = self.api_key
                # Update the API key in various places

                filename = os.path.join(os.path.dirname(__file__), 'apikey.json')
                print(f"Writing API key to: {filename}")
                # Decide where to save the API key on disk

                with open(filename, 'w') as file:
                    json.dump({'api_key': new_key}, file)
                # Write the new API key to disk

                api_key_window.destroy()
                # Close the window after the new API key is set

        api_key_window = tk.Toplevel(self.root)
        api_key_window.title("API Key")
        # Create a new top-level window

        label = tk.Label(api_key_window, text="Enter new API Key: ", font=(self.font_family, self.font_size))
        label.pack(padx=10, pady=10)
        # Add a label to the window

        entry = tk.Entry(api_key_window, font=(self.font_family, self.font_size))
        entry.pack(padx=10, pady=5)
        # Add an Entry widget (text field) to the window

        set_api_key_button = tk.Button(api_key_window, text="Set API Key", command=on_set_api_key,
                                       font=(self.font_family, self.font_size))
        set_api_key_button.pack(padx=10, pady=10)
        # Add a button to the window, which will call the on_set_api_key function when clicked

        self.center_window2(api_key_window)
        # Center the window on the screen

        api_key_window.wait_window()  # Block until the window is destroyed
        # Pause the program until the window is closed

    def center_window(self, window):
        window_width = 760
        window_height = 620
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 3) - (window_height // 2)
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def center_window2(self, window):
        window_width = 400
        window_height = 150
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 3) - (window_height // 2)
        window.geometry(f"{window_width}x{window_height}+{x}+{y}")

    def get_gpt_response(self, user_prompt):
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + openai.api_key,
        }

        self.conversation_history.append({'role': 'user', 'content': user_prompt})

        data = {
            'model': 'gpt-4-1106-preview',
            # 'model': 'gpt-3.5-turbo-16k',
            'messages': self.conversation_history,
            'temperature': 0.7,
            'top_p': 0.9,
            'presence_penalty': 0.6,
            'frequency_penalty': 0.3,
        }

        response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, data=json.dumps(data))

        if self.loop_active and self.loop_text:  # If loop is active and loop text is not None
            self.root.after(1000, self.loop_request)  # run the loop request after 500ms (0.5 second)

        return response

    def loop(self):
        if not self.loop_active:
            self.loop_dialog()

    def loop_dialog(self):
        def on_loop():
            self.loop_active = True
            self.loop_text = entry.get()
            loop_window.destroy()
            self.loop_request()

        loop_window = tk.Toplevel(self.root)
        loop_window.title("Loop")

        label = tk.Label(loop_window, text="Enter response text to loop:\n", font=(self.font_family, self.font_size))
        label.pack(padx=10, pady=10)

        bold_font = font.Font(label, label.cget("font"))
        bold_font.configure(weight="bold")

        warning_label = tk.Label(loop_window, text="WARNING:", font=bold_font, compound="left")
        warning_label.pack()

        rest_of_text = tk.Label(loop_window, text="This will auto-loop your prompt until stopped!",
                                font=(self.font_family, self.font_size))
        rest_of_text.pack()

        entry = tk.Entry(loop_window, font=(self.font_family, self.font_size), width=50)
        entry.pack(padx=10, pady=5)

        loop_button = tk.Button(loop_window, text="Loop", command=on_loop, font=(self.font_family, self.font_size))
        loop_button.pack(padx=10, pady=10)

        # Set the font for the labels and button
        label.configure(font=(self.font_family, self.font_size))
        entry.configure(font=(self.font_family, self.font_size))
        loop_button.configure(font=(self.font_family, self.font_size))

        # Set the geometry of the loop window
        window_width = 450  # Change this value as desired
        window_height = 220  # Change this value as desired
        screen_width = loop_window.winfo_screenwidth()
        screen_height = loop_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 3
        loop_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        loop_window.resizable(False, False)

        loop_window.transient(self.root)
        loop_window.grab_set()
        self.root.wait_window(loop_window)

    def cancel_loop(self):
        self.loop_active = False
        self.loop_text = None

    def loop_request(self):
        if self.loop_active and self.loop_text:
            self.send_message(auto=True)

    def send_message(self, event=None, auto=False):
        if not auto:
            user_message = self.text_input.get()
        else:
            user_message = self.loop_text

        if user_message == 'exit':
            if messagebox.askokcancel("Quit", "Do you really want to quit?"):
                self.root.destroy()
        else:
            self.text_input.set('')
            self.conversation.config(state='normal')  # Enable editing
            self.conversation.insert(tk.END, "You: ", 'bold-text')
            self.conversation.insert(tk.END, user_message + '\n\n', 'red-text')
            self.conversation_history.append({'role': 'user', 'content': user_message})
            self.conversation.config(state='disabled')  # Disable editing

            # Change the line below to set your desired hex color
            self.conversation.tag_configure('red-text', foreground='#b00707')  # Replace '#FF0000' with your hex color

            def gpt_request():
                response = self.get_gpt_response(user_message)
                if response.status_code == 200:
                    completion = response.json()['choices'][0]['message']['content']
                    self.conversation.config(state='normal')  # Enable editing
                    self.conversation.insert(tk.END, "JChat: ", 'bold-text')
                    self.conversation.insert(tk.END, completion + '\n\n', 'blue-text')
                    self.conversation_history.append({'role': 'assistant', 'content': completion})
                    self.conversation.insert(tk.END, '_' * 80, 'line')  # Add a visual line break
                    self.conversation.insert(tk.END, '\n\n')
                    self.conversation.config(state='disabled')  # Disable editing
                    self.conversation.see(tk.END)

                    # Perform another command with the completion here
                    # Example: Call another function with completion as an argument
                    # another_function(completion)...

                else:
                    print("An error occurred:", response.text)

            # Change the line below to set your desired hex color
            self.conversation.tag_configure('blue-text', foreground='#0707b0')  # Replace '#FF0000' with your hex color

            # Configure the visual line tag
            self.conversation.tag_configure('line', underline=True)

            # Configure the bold text tag
            self.conversation.tag_configure('bold-text', font=(self.font_family, self.font_size, 'bold'))

            threading.Thread(target=gpt_request).start()

    def clear_conversation(self):
        confirmed = messagebox.askyesno("Clear Conversation", "Are you sure you want to clear the conversation?")
        if confirmed:
            self.conversation.config(state='normal')  # Enable editing
            self.conversation.delete('1.0', tk.END)
            self.conversation.config(state='disabled')  # Disable editing

    def exit_app(self):
        self.root.destroy()

    def change_behavior(self):
        def select_behavior(behavior):
            self.pre_prompt = self.behaviors[behavior]
            self.conversation_history = [{'role': 'system', 'content': self.pre_prompt}]
            window.destroy()

        window = tk.Toplevel(self.root)
        window.title("Select Behavior")

        buttons = [tk.Button(window, text=name, command=lambda name=name: select_behavior(name)) for name in
                   self.behaviors.keys()]

        rows = round(len(buttons) ** 0.5)
        cols = len(buttons) // rows + (len(buttons) % rows > 0)

        for i, button in enumerate(buttons):
            button.grid(row=i // cols, column=i % cols, padx=10, pady=10, sticky="we")

            # Set the font for the button
            button.configure(font=(self.font_family, self.font_size))

            # Center the window on the screen
            window_width = window.winfo_reqwidth()
            window_height = window.winfo_reqheight()
            screen_width = window.winfo_screenwidth()
            screen_height = window.winfo_screenheight()
            x = (screen_width // 2) - (window_width // 2)
            y = (screen_height // 3) - (window_height // 2)
            window.geometry(f"+{x}+{y}")

    def run(self):
        self.root.mainloop()
