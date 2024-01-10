import errno
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import PhotoImage

import customtkinter
import pyperclip
from customtkinter import CTkFont
from openai import OpenAI

CUSTOM_PROMPT_FILE_NAME = ".custom_prompt.txt"
CLIPBOARD_PLACEHOLDER = "{CLIPBOARD}"

client = OpenAI()

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

#
default_model = 'gpt-4-1106-preview'


def app_help():
    print("Usage:")
    print(" ", sys.argv[0], "<ACTION>")
    print(" Supported actions: Rewrite, Ask, CustomPrompt")


class App(customtkinter.CTk):
    MAX_SIZE = 1000

    def __init__(self):
        super().__init__(className="AI Helper")
        self.app_path = Path(__file__).resolve().parent

        self.SUPPORTED_ACTIONS = {
            "Rewrite": self.execute_rewrite,
            "Ask": self.execute_ask_question,
            "CustomPrompt": self.execute_custom_prompt
        }

        if len(sys.argv) < 2:
            print("Missing parameter ACTION")
            app_help()
            sys.exit(errno.EPERM)

        self.action = sys.argv[1]
        if self.SUPPORTED_ACTIONS.get(self.action) is None:
            print('Unsupported action:', self.action)
            app_help()
            sys.exit(errno.EPERM)

        # UI
        monospace_font = CTkFont(family="monospace", size=16, weight="normal")

        # configure window
        if self.action == 'Rewrite':
            self.title("AI Rewriter")
        else:
            self.title("AI Helper")

        self.geometry(f"{1000}x{700}")
        self.iconphoto(False, PhotoImage(file=self.app_path / "assets/app-icon.png"))

        # configure grid layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.textbox_question = customtkinter.CTkTextbox(self, font=monospace_font, height=150)
        self.textbox_question.grid(row=0, column=0, columnspan=2, sticky="nsew")

        # Question button
        question_button_title = self.action
        if self.action == 'CustomPrompt':
            question_button_title = 'Execute custom prompt'

        self.answer_button = customtkinter.CTkButton(master=self, text=question_button_title,
                                                     command=self.answer_button_event)
        self.answer_button.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Label
        self.info_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=14, weight="bold"))
        self.info_label.grid(row=1, column=1, padx=20, pady=(5, 5))

        # Answer textbox
        self.textbox_answer = customtkinter.CTkTextbox(self, font=monospace_font)
        self.textbox_answer.grid(row=2, column=0, columnspan=2, sticky="nsew")

        # Initialize
        self.user_input = pyperclip.paste()

        if self.action == 'Ask':
            self.user_input = 'Explain: ' + self.user_input

        if self.action == 'CustomPrompt':
            self.textbox_question.insert("0.0", self.get_custom_prompt())
        else:
            self.textbox_question.insert("0.0", self.user_input)

        self.textbox_question.focus_set()

        # Bind keyboard shortcuts
        self.textbox_question.bind(
            '<Control_L><Return>',
            # Return value "break" means that the event should not be processed by the default handler
            lambda event: "break" if self.answer_button_event() is None else None
        )
        self.textbox_question.bind('<Escape>', lambda event: self.quit())
        self.textbox_answer.bind('<Escape>', lambda event: self.quit())

        # Initial execution at the startup
        if self.action == 'Rewrite' and self.user_input is not None:
            self.answer_button_event()
        if self.action == 'CustomPrompt' and self.user_input is not None:
            self.answer_button_event()

    def answer_button_event(self):
        self.set_working_state('Let me think...')
        self.execute_in_thread(lambda: self.SUPPORTED_ACTIONS[self.action](
            self.clip_text(str(self.textbox_question.get("0.0", "end")), self.MAX_SIZE)), ())

    def set_working_state(self, message):
        self.answer_button.configure(state=customtkinter.DISABLED, fg_color="#BB6464")
        self.info_label.configure(text=message)

    def unset_working_state(self, message):
        self.answer_button.configure(state=customtkinter.NORMAL, fg_color=["#3B8ED0", "#1F6AA5"])
        self.info_label.configure(text=message)

    def quit(self):
        self.destroy()

    def execute_rewrite(self, text_to_rewrite):
        try:
            # Execute the prompt
            prompt = f"Please rewrite the following text for more clarity and make it grammatically correct. Give me the " \
                     f"updated text. The updated text should be correct grammatically and stylistically and should be " \
                     f"easy to follow and understand. Only make a change if it's needed. Try to follow the style of the " \
                     f"original text. " \
                     f"Don't make it too formal. Include only improved text no other " \
                     f"commentary.\n\nThe text to check:\n---\n{text_to_rewrite}\n---\n\nImproved text: "

            completion = client.chat.completions.create(
                model=default_model, temperature=1,
                messages=[{"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content

            self.textbox_answer.delete("0.0", "end")
            self.textbox_answer.insert("0.0", result)

            pyperclip.copy(result)
            self.unset_working_state('Copied to clipboard')

            self.log_to_file('Rewrite', text_to_rewrite, result)
        except Exception as e:
            self.info_label.configure(text='Oops, something went wrong. Try again later.')
            self.unset_working_state('')

    def execute_ask_question(self, question):
        try:
            # Execute the prompt
            completion = client.chat.completions.create(
                model=default_model, temperature=0,
                messages=[{"role": "user", "content": question}]
            )
            result = completion.choices[0].message.content

            self.textbox_answer.delete("0.0", "end")
            self.textbox_answer.insert("0.0", result)

            self.info_label.configure(text='')
            self.unset_working_state('')

            self.log_to_file('Question', question, result)
        except Exception as e:
            self.info_label.configure(text='Oops, something went wrong. Try again later.')
            self.unset_working_state('')

    def execute_custom_prompt(self, question):
        try:
            self.update_custom_prompt(question)

            # Execute the prompt
            custom_prompt = self.get_custom_prompt()
            prompt = self.render_custom_prompt(custom_prompt, pyperclip.paste())
            print(prompt)
            completion = client.chat.completions.create(
                model=default_model, temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            result = completion.choices[0].message.content

            self.textbox_answer.delete("0.0", "end")
            self.textbox_answer.insert("0.0", result)

            self.info_label.configure(text='')
            self.unset_working_state('')

            self.log_to_file('CustomPrompt', prompt, result)
        except Exception as e:
            self.info_label.configure(text='Oops, something went wrong. Try again later.')
            self.unset_working_state('')

    def log_to_file(self, input_type, content, answer):
        log_file = self.app_path / "ai_helper.log"
        with open(log_file, 'a') as f:
            current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f'Date: {current_date}\n')
            f.write(f'{input_type}: {content}\n')
            f.write(f'Answer: {answer}\n---\n')

    def get_custom_prompt(self):
        custom_prompt_file = self.app_path / CUSTOM_PROMPT_FILE_NAME

        if not os.path.exists(custom_prompt_file):
            with open(custom_prompt_file, 'w') as f:
                f.write(
                    f"Create a concise summary of the following text:\n\n"
                    f"```\n"
                    f"{CLIPBOARD_PLACEHOLDER}\n"
                    f"```")

        with open(custom_prompt_file, 'r') as file:
            return file.read()

    def update_custom_prompt(self, new_prompt):
        custom_prompt_file = self.app_path / CUSTOM_PROMPT_FILE_NAME

        with open(custom_prompt_file, 'w') as f:
            if CLIPBOARD_PLACEHOLDER not in new_prompt:
                new_prompt = new_prompt + '\n\n' + CLIPBOARD_PLACEHOLDER
            f.write(new_prompt)

    def render_custom_prompt(self, prompt, text):
        return prompt.replace(CLIPBOARD_PLACEHOLDER, text)

    @staticmethod
    def clip_text(text, max_size):
        if text is None:
            return None

        if len(text) > max_size:
            return text[:max_size].strip()
        else:
            return text.strip()

    @staticmethod
    def execute_in_thread(callback, args):
        thread = threading.Thread(target=callback, args=args)
        thread.start()


if __name__ == "__main__":
    app = App()
    app.mainloop()
