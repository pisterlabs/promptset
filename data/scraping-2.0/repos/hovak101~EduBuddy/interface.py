import tkinter as tk
from screeninfo import get_monitors
from PIL import Image, ImageTk
import os
from tkinter import filedialog
import TextConverter as tc
from tkinter import messagebox
import platform
import pyperclip
import config
from threading import Thread
from Speech_functions import checking, asking
import textwrap
import time
from llama_index import VectorStoreIndex, SimpleDirectoryReader, GPTVectorStoreIndex
from langchain.memory import ConversationSummaryMemory
import llama_index
import re
import openai
import json
from tkinter import ttk
config.init()
from langchain.chat_models import ChatOpenAI
import pickle
# import ctypes
# import objc

"""
Changes to make:
- icons for all buttons
- rounded corners
- smooth animation for expanding/compressing window
- Change all text in window based on language
- Document for other files
"""

def print_function_name(func):
    def wrapper(*args, **kwargs):
        print(f'Executing {func.__name__}')
        return func(*args, **kwargs)
    return wrapper


class Quiz:
    """Quiz object for iterate questions"""
    #@print_function_name
    def __init__(self, quiz_input_string, num_quiz_questions = 5):
        self.questions = [None for _ in range(num_quiz_questions)]
        lines = quiz_input_string.split("\n")
        for i in range(num_quiz_questions):
            self.questions[i] = {
                "question": lines[i * 6][3:],
                "alternatives": ["", "", "", ""],
                "answer": -1,
            }
            for j in range(4):
                init_string = lines[i * 6 + j + 1][3:]
                asterisk_index = init_string.find("*")

                # Create the substring based on the asterisk index
                if asterisk_index != -1:
                    init_string = init_string[:asterisk_index]
                    self.questions[i]["answer"] = j

                self.questions[i]["alternatives"][j] = init_string

        # self.questions is formatted like this: obj = [{question: "<q>", alternatives: ["alt1", "alt2", "alt3", "alt4"], answer: <0-3>}]

class Window(tk.Tk):
    """Main window"""
    NUM_QUIZ_QUESTIONS = 5
    JSON_NAME = 'EduBuddy_Memory.json'
    PICKLE_NAME = 'EduBuddy_Memory.pkl'
    #@print_function_name
    def __init__(self, threads : list):
        super().__init__()
        self.end = False
        self.configure(bg = "white")
        self.threads = threads
        self.context = ""
        self.is_left = False
        self.is_up = False
        # Check windows
        llm = ChatOpenAI(model_name = "gpt-4", temperature = 0.9)
        if os.path.exists(Window.PICKLE_NAME):
            with open(Window.PICKLE_NAME, 'rb') as f:
                self.memory = pickle.load(f)
        else:
            self.memory = ConversationSummaryMemory(llm = llm)
        # if os.path.exists(Window.JSON_NAME):
        #     with open(Window.JSON_NAME, 'r') as f:
        #         memory = json.load(f)
        #         self.memory.save_context({"input": f"Here is the context from old conversation {memory['history']}"}, {"output": "Okay, I will remember those!"})
        self.subtractedDistace = 25
        self.addedDistance = 25
        if (platform.system()) == "Windows":
            self.addedDistance = 80
            
        self.save = ""
        self.title("EduBuddy")
        self.before_text = 0
        # self.overrideredirect(True)  # Remove window decorations (title, borders, exit & minimize buttons)
        self.attributes("-topmost", True)
        self.messagebox_opening = False
        self.quiz_opening = False
        # screen info
        screen = get_monitors()[0]  # number can be changed ig
        self.screen_w = screen.width
        self.screen_h = screen.height
        self.is_maximized = False

        # Set the window's initial position
        self.padding_w = int(self.screen_w * 0.005)
        self.padding_h = int(self.screen_w * 0.005)
        self.sq_button_height = 45
        self.language = tk.StringVar(self)
        self.language.set("English")
        # summarize, erase, show, save, quiz, close, microphone, file, text button and textbox
        self.summarize_button = AButton(self, text = "Summarize", command = self.summarize_button_press)
        self.erase_button = AButton(self, text = "Erase", command = self.erase_button_press)
        self.show_button = AButton(self, text = "Show", command = self.show_button_press)
        self.save_button = AButton(self, text = "Save", command = self.save_button_press)
        self.quiz_button = AButton(self, text = "Quiz", command = self.quiz_button_press)
        self.language_button = tk.OptionMenu(self, self.language, "English", "Italian", "Afrikaans", "Spanish", "German", "French", "Indonesian", "Russian", "Polish", "Ukranian", "Greek", "Latvian", "Mandarin", "Arabic", "Turkish", "Japanese", "Swahili", "Welsh", "Korean", "Icelandic", "Bengali", "Urdu", "Nepali", "Thai", "Punjabi", "Marathi", "Telugu")#AButton(self, text = "Language", command = self.language_button_press)
        self.mic_button = AButton(self, text = "From Mic", command = asking)
        self.file_button = AButton(self, text = "From File", command = self.file_button_press)
        self.text_button = AButton(self, text = "From Text", command = self.text_button_press)
        self.context_title = tk.Label(self, text = "Context", bg = "lightblue")
        self.minimize_button = AButton(self, text = '-', command = self.minimize_button_press)
        self.maximize_button = AButton(self, text = '+', command = self.maximize_button_press)
        self.close_button = AButton(self, text = "x", command = self.close_button_press)
        
        self.icon_size = 45
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, "media", "buddy.png")
        self.image = Image.open(image_path)
        self.image = self.image.resize((self.icon_size, self.icon_size))
        self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        self.was_right = True
        self.image_tk = ImageTk.PhotoImage(self.image)
        self.img_label = tk.Label(self, image = self.image_tk)

        # Text output
        self.output_box = tk.Text(self, borderwidth = 0, highlightthickness = 0, font = ("Times New Roman", 14))
        self.change_size(w = 400, h = 500)
        self.output_box.configure(state = "normal")
        # # Text input field
        self.output_box.delete("1.0", tk.END)
        # self.output_box.bind("<Return>", self.text_button_press)

        # Bind mouse events
        self.bind("<ButtonPress-1>", self.on_button_press)
        self.bind("<B1-Motion>", self.on_button_motion)
        self.bind("<ButtonRelease-1>", self.on_button_release)
        
        # Quiz variables
        self.current_quiz_ans = -1
        self.current_quiz_score = 0
        self.current_quiz_questions = []
        self.quiz_obj = None
        self.quiz_alternative_buttons = [None, None, None, None]

    #@print_function_name
    def maximize_button_press(self):
        """Maximize window"""
        if not self.is_maximized:
            self.is_maximized = True
            self.info = (self.is_left, self.is_up, self.w ,self.h)
            self.is_left = True
            self.is_up = True
            self.change_size(w = self.screen_w - 2 * self.padding_w, h = self.screen_h - 2 * self.padding_h- 25, changed = not self.info[0])
        else:
            self.is_maximized = False
            (self.is_left, self.is_up, w ,h) = self.info
            self.change_size(w = w, h = h, changed = not self.is_left)

    #@print_function_name
    def minimize_button_press(self):
        """Minimize window"""
        self.messagebox_opening = True
        messagebox.showwarning(title = "Minimize warning", message = "Be careful, there will be error if you are using Stage Manager on Mac")
        self.messagebox_opening = False
        self.overrideredirect(False)
        self.wm_state('iconic')

    #@print_function_name
    def change_size(self, w = None, h = None, changed = None):
        """Change size of window, and position of elements if needed"""
        if w is not None:
            self.w = w  # was 200
        if h is not None:
            self.h = h  # was 300

        # self.x = self.screen_w - self.w - self.padding_w  # X coordinate
        # self.y = self.screen_h - self.h - self.padding_h  # Y coordinate

        self.x = self.padding_w if self.is_left else self.screen_w - self.w - self.padding_w
        self.y = self.padding_h + self.addedDistance if self.is_up else self.screen_h - self.h - self.padding_h - self.subtractedDistace #- self.addedDistance
        if changed:
            self.img_label.destroy()
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
            self.image_tk = ImageTk.PhotoImage(self.image)
            self.img_label = tk.Label(self, image = self.image_tk)
            self.was_right = not self.was_right

        self.geometry(f"+{self.x}+{self.y}")
        if w is not None or h is not None:
            self.geometry(f"{self.w}x{self.h}")

        # summarize button
        self.summarize_button.place(x = 0, y = 0, width = self.w / 5, height = self.sq_button_height)
        
        # erase the screen
        self.erase_button.place(x = self.w / 5, y = 0, width = self.w / 5, height = self.sq_button_height)

        # show memory
        self.show_button.place(x = self.w * 2 / 5, y = 0, width = self.w / 5, height = self.sq_button_height)

        # save memory
        self.save_button.place(x = self.w * 3 / 5, y = 0, width = self.w / 5, height = self.sq_button_height)

        # quiz button
        self.quiz_button.place(x = self.w * 4 / 5, y = 0, width = self.w / 5, height = self.sq_button_height)

        # close button
        # self.language_button.place(x = 0, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)

        # button get from microphone
        self.mic_button.place(x = self.w / 5, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)

        # button get from local file
        self.file_button.place(x = self.w * 2 / 5, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)

        # button get from text
        self.text_button.place(x = self.w * 3 / 5, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)

        # button minimize
        # self.maximize_button.place(x = -17.5 + (self.w - self.icon_size + self.w * 4 / 5) / 2, y = self.h - 50, width = 35, height = self.sq_button_height / 3)
        # self.minimize_button.place(x = -17.5 + (self.w - self.icon_size + self.w * 4 / 5) / 2, y = self.h - 35, width = 35, height = self.sq_button_height / 3)
        # self.close_button.place(x = -17.5 + (self.w - self.icon_size + self.w * 4 / 5) / 2, y = self.h - 20, width = 35, height = self.sq_button_height / 3)

        # Context title box
        self.context_title.place(x = 3, y = 45, w = self.w - 6, h = 25)

        # self.img_label.place(x = self.w - self.icon_size, y = self.h - self.icon_size)

        self.output_box.place(x = 3, y = 65, w = self.w - 6, h = (self.h - 2 * self.sq_button_height - 25), )
        # self.output_box.config(highlightbackground = 'black', highlightthickness = 1)
        if self.is_left:
            self.img_label.place(x = 5, y = self.h - self.icon_size - 5)
            self.language_button.place(x = self.w * 4 / 5, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)
        else:
            self.img_label.place(x = self.w - self.icon_size - 5, y = self.h - self.icon_size - 5)
            self.language_button.place(x = 0, y = self.h - 50, width = self.w / 5, height = self.sq_button_height)

    #@print_function_name
    def close_button_press(self):
        """Close window with message box"""
        self.messagebox_opening = True
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.end = True
            for t in self.threads:
                t.join()
            # with open(Window.JSON_NAME, 'w') as f:
            #     json.dump(self.memory.load_memory_variables({}), f)
            with open(Window.PICKLE_NAME, 'wb') as f:
                pickle.dump(self.memory, f)
            self.destroy()
        self.messagebox_opening = False

    #@print_function_name
    def file_button_press(self):
        """Open file(s) to query"""
        self.context_title.config(text = "Read from file(s)")
        self.output_box.configure(state = "disabled")
        file_path = filedialog.askopenfilenames(
            parent = self, title = "Choose one or multiple file(s)"
        )
        self.output_box.configure(state = "normal")
        if len(file_path) != 0:
            # Do something with the selected file path, such as printing it
            documents = SimpleDirectoryReader(input_files = file_path).load_data()
            # index = VectorStoreIndex.from_documents(documents)
            index = GPTVectorStoreIndex.from_documents(documents)
            # query index
            query_engine = index.as_query_engine()
            self.context_title.config(
                text = "Enter your question about the file anywhere below"
            )
            summary = query_engine.query("Summarize key informations in this/ these files!")
            # print("\n", summary, end = "\n\n")
            self.output_box.insert(tk.END, f"\n{summary.response}\n")
            self.save += "(Summary from documents: " + str(summary.response) + "), "
            # response = query_engine.query("Explain me The Schrodinger equation")
            # result = query_engine.query("Why do we need quantum mechanics")
            # answer = query_engine.query("Who is Julia Cook")
            # random = query_engine.query("Who is Leo Messi")
            # print("Count:", index., "\n\n\n\n")
            # for doc_id in index.document_ids():
            #     embedding = index.embedding_for_document(doc_id)
            #     print(f"Embedding for document {doc_id}: {embedding}")
            # print("\n", response, end = "\n\n")
            # print("\n", result, end = "\n\n")
            # print("\n", answer, end = "\n\n")
            # print("\n", random, end = "\n\n")
            # # print("Selected file:", file_path)
            # print(len(documents))

    #@print_function_name
    def in_textbox(self, x, y, xx, yy):
        """Return true only if the position of mouse is in textbox"""
        x1, y1, w, h = 0, 30, self.w - 6, (self.h - 2 * self.sq_button_height - 25)
        x2, y2 = x1 + w, y1 + h
        return x1 <= x <= x2 and y1 <= y <= y2

    #@print_function_name
    def on_button_press(self, event):
        """Track button press"""
        if self.in_textbox(event.x, event.y, event.x_root, event.y_root):
            if not self.messagebox_opening and not self.quiz_opening:
                self.messagebox_opening = True
                # print("before:", self.is_up)
                self.change_size(w = 600, h = 750)
                self.output_box.config(font = ("Times New Roman", 21))
                # self.output_box.configure(state = 'disabled')
                self.output_box.configure(state = 'normal')
                # self.output_box.insert(tk.END, "HEY!\n")

        else:
                self.change_size(w = 400, h = 500)
                self.output_box.config(font = ("Times New Roman", 14))
                # Capture the initial mouse position and window position
                self.x = event.x_root
                self.y = event.y_root
                self.offset_x = self.winfo_x()
                self.offset_y = self.winfo_y()
                self.messagebox_opening = False
                self.output_box.configure(state = 'disabled')
                # self.output_box.configure(state = 'normal')
                # self.output_box.insert(tk.END, "HEY!\n")

    #@print_function_name
    def on_button_motion(self, event):
        """Move window with the mouse if it holds"""
        if not self.messagebox_opening and not self.in_textbox(event.x, event.y, event.x_root, event.y_root):
            # Calculate the new window position based on mouse movement
            new_x = self.offset_x + (event.x_root - self.x)
            new_y = self.offset_y + (event.y_root - self.y)
            self.geometry(f"+{new_x}+{new_y}")

    #@print_function_name
    def on_button_release(self, event):
        """Stick to closest corner when release"""
        if not self.messagebox_opening and not self.in_textbox(event.x, event.y, event.x_root, event.y_root):
            changed = self.is_left != (event.x_root - event.x + self.w / 2 < self.screen_w / 2)
            self.is_left = event.x_root - event.x < (self.screen_w - self.w) / 2
            self.is_up = event.y_root - event.y < (self.screen_h - self.h) / 2
            self.change_size(changed = changed)

    #@print_function_name
    def waitAndReturnNewText(self):
        """Running in background waiting for pasteboard"""
        while not self.end:
            try:
                config.text = pyperclip.waitForNewPaste(timeout = 10)
            except:
                pass
    
    #@print_function_name
    def summarize_button_press(self):
        """Summarize text in pasteboard"""
        # self.output_box.configure(state = "disabled")
        # Destroy old canvas
        try:
            self.canvas.destroy()
        except:
            pass
        text = ' '.join(re.split(" \t\n", config.text))
        if text != "":
            if len(text.split(" ")) >= 30:
                # generate title
                title = tc.getTitleFromText(text, self.language.get())
                self.context_title.config(
                    text = textwrap.fill(title.split('"')[1], width = self.w - 20)
                )
                # generate summary
                minimumWords = 0
                maximumWords = tc.getResponseLengthFromText(text)
                response = self.run_gpt(tc.generateSummaryFromText, (text, minimumWords, maximumWords, self.language.get()))
                # thread = Thread(target = window.waitAndReturnNewText)
                # thread.start()
                # self.threads.append(thread)
                # self.output_box.configure(state = "normal")
                self.output_box.insert(tk.END, f"\nSummary:\n{response}\n")
                self.before_text = len(self.output_box.get("1.0", tk.END))
                self.save += "(Summary: " + response + "), "
            else:
                # self.output_box.configure(state = "normal")
                self.output_box.insert(tk.END, "\nPlease choose a longer text to summarize\n")
        else:
            # self.output_box.configure(state = "normal")
            self.output_box.insert(tk.END, "\nNo text found! Choose a new text if this keep happens\n")
        # self.output_box.configure(state = 'normal')
        # print(self.messagebox_opening)

    #@print_function_name
    def quiz_button_press(self):
        """Generate quizzes from pasteboard"""
        # generate title
        self.messagebox_opening = True
        self.quiz_opening = True
        print(self.output_box.get("1.0", tk.END))
        # self.geometry("600x750")
        if messagebox.askyesno("Quiz", "Are you sure you are ready for the quiz? Also, if you want to save this conversation, click cancel and click 'Save'"):
            self.messagebox_opening = False
            self.output_box.delete("1.0", tk.END)
            self.output_box.configure(state = "disabled")
            self.geometry("800x1200")
            text = ' '.join(re.split(" \t\n", config.text))
            print(len(text), text[:100], )
            if text != "":
                if len(text.split(" ")) >= 50:
                    title = tc.getTitleFromText(text, self.language.get())
                    self.context_title.config(
                        text = textwrap.fill(title.split('"')[1], width = self.w - 20)
                    )
                    # generate quiz
                    response = self.run_gpt(tc.getMultipleChoiceQuiz, (text, self.language.get(), 5))
                    self.quiz_obj = Quiz(response, Window.NUM_QUIZ_QUESTIONS)
                    self.quiz_iteration(self.quiz_obj)
                else:
                    self.context_title.config(
                        text = "Please choose a longer text to make quiz"
                    )
            else:
                self.context_title.config(
                    text = "No text found! Choose a new text if this keep happens"
                )
            self.output_box.configure(state = "normal")
        else:
            self.messagebox_opening = False
            self.quiz_opening = False

    #@print_function_name
    def show_button_press(self):
        """Show memory (saved and unsaved)"""
        self.messagebox_opening = True
        new_window = tk.Toplevel(self)
        new_window.title("Memory")
        t = tk.Text(new_window, borderwidth = 0, highlightthickness = 0)
        t.pack()
        t.insert(tk.END, f"Unsaved: {self.save}\nSaved: {self.memory.load_memory_variables({})['history']}")
        t.configure(state = "disabled")
        new_window.grab_set()
        self.wait_window(new_window)
        self.messagebox_opening = False

    #@print_function_name
    def text_button_press(self):
        """Answer to text inside textbox from user"""
        text = ' '.join(re.split(" \t\n", self.output_box.get("1.0", "end-1c")[max(0, self.before_text-1):]))
        
        if len(text) >= 2:
            str1 = self.run_gpt(tc.sendGptRequest, (text, config.text, self.language.get(), self.memory))
            try:
                output ='\n'.join(str1.split('\n\n')[1:])
                self.save += "(Q: " + text + " and A: " + str1 + "), "
                if output == '':
                    raise ValueError
            except:
                output = str1
            self.output_box.insert(tk.END, '\n\n' + output + '\n')
            self.before_text = len(self.output_box.get("1.0", tk.END))
            return 'break'
            # Run your function here. And then with the gpt output, run insert it into output box
        else:
            self.context_title.config(
                text = "Your text is too short to do any work!"
            )
    
    #@print_function_name
    def quiz_iteration(self, quiz_obj):
        """Iterate through questions in quiz generated and put it nicely in canvas"""
        if len(quiz_obj.questions) == 0:
            self.canvas.destroy()
            self.display_quiz_results()
            return

        # Destroy old canvas
        try:
            self.canvas.destroy()
        except:
            pass

        # make quiz question and button element from Quiz obj
        self.canvas = tk.Canvas(self, width = self.w, height = 300)
        wrapped_text = textwrap.fill(
            quiz_obj.questions[0]["question"], width = self.w - 20
        )
        self.question = self.canvas.create_text(
            self.w // 2, 30, text = wrapped_text, width = self.w - 40
        )
        self.quiz_alternative_buttons = []
        for i in range(4):
            x1, y1, x2, y2 = 10, 65 + i * 45, self.w - 10, 110 + i * 45
            rect = self.canvas.create_rectangle(x1, y1, x2, y2, fill = "white")
            text = self.canvas.create_text(
                (x1 + x2) // 2,
                (y1 + y2) // 2,
                text = textwrap.fill(
                    f"""{i+1}. {quiz_obj.questions[0]["alternatives"][i]}""",
                    width = self.w - 20,
                ),
                width = self.w - 40,
            )
            self.canvas.tag_bind(
                rect,
                "<Button-1>",
                lambda event, choice = i: self.quiz_choice(event, choice),
            )
            self.canvas.tag_bind(
                text,
                "<Button-1>",
                lambda event, choice = i: self.quiz_choice(event, choice),
            )
            self.quiz_alternative_buttons.append((rect, text))

        self.current_quiz_ans = quiz_obj.questions[0]["answer"]
        self.current_quiz_questions.append([wrapped_text])
        quiz_obj.questions.pop(0)
        self.canvas.place(x = 0, y = (-100 + 45 * (i + 1)), w = self.w, h = 300)

    #@print_function_name
    def quiz_choice(self, event, choice):
        """Response to users' choices"""
        if choice == self.current_quiz_ans:
            self.current_quiz_score += 1
        for rect, text in self.quiz_alternative_buttons:
            self.canvas.itemconfig(rect, fill = "white")
        self.canvas.itemconfig(self.quiz_alternative_buttons[choice][0], fill = "red")
        self.canvas.itemconfig(
            self.quiz_alternative_buttons[self.current_quiz_ans][0], fill = "green"
        )
        self.current_quiz_questions[-1].append(
            self.canvas.itemcget(self.quiz_alternative_buttons[choice][1], "text")
            .strip()
            .split(maxsplit = 1)[1]
        )
        self.current_quiz_questions[-1].append(
            self.canvas.itemcget(
                self.quiz_alternative_buttons[self.current_quiz_ans][1], "text"
            )
            .strip()
            .split(maxsplit = 1)[1]
        )
        self.after(ms = 2000, func = lambda: self.quiz_iteration(self.quiz_obj))

    #@print_function_name
    def display_quiz_results(self):
        """Display quiz results"""
        output = (
            f"Quiz results: {self.current_quiz_score}/{Window.NUM_QUIZ_QUESTIONS}:\n\n"
        )
        for id, vals in enumerate(self.current_quiz_questions):
            try:
                output += f"Question {id + 1}: {vals[0]}\nResult: {'Correct' if vals[1] == vals[2] else 'Incorrect'}!\nYour choice: {vals[1]}\nAnswer: {vals[2]}\n\n"
            except:
                pass
        self.save += "(Quiz:" + ' '.join(re.split(" \t\n", str(self.current_quiz_questions))) + "), "
        self.output_box.insert(tk.END, f"\n{output}")
        self.before_text = len(self.output_box.get("1.0", tk.END))
        self.quiz_opening = False

    #@print_function_name
    def save_button_press(self):
        """Save unsaved memory to saved memory to later save into file"""
        self.output_box.delete("1.0", tk.END)
        self.memory.save_context({"input": f"""Here is a context (remember topic and user's info) for future requests: {self.save}"""},
                                         {"output": f"""Thank you, I will remember and be here for you!"""})
        self.save = ""

    #@print_function_name
    def load_data(self, func, val, ret):
        """Run function and value set from run_gpt function"""
        ret[0] = func(*val)

    #@print_function_name
    def run_gpt(self, func, val):
        """Run complicated functions in another thread"""
        ret = [" "]
        loading_window = LoadingWindow(self, ret)
        thread = Thread(target = self.load_data, args = (func, val, ret))
        thread.start()
        loading_window.grab_set()
        self.wait_window(loading_window)
        return ret[0]
    
    #@print_function_name
    def erase_button_press(self):
        """Erase all memory (saved and unsaved and in file)"""
        llm = ChatOpenAI(model_name = "gpt-4", temperature = 0.9)
        self.memory = ConversationSummaryMemory(llm = llm)
        with open(Window.PICKLE_NAME, 'wb') as f:
            pickle.dump(self.memory, f)
        self.save = ""

class LoadingWindow(tk.Toplevel):
    """Loading window to let user know the system is running"""
    #@print_function_name
    def __init__(self, master, ret):
        super().__init__(master)
        self.ret = ret
        self.title("Loading")
        self.string = tk.StringVar(self, "Working on it")
        label = tk.Label(self, textvariable = self.string)
        label.pack()
        self.progress = ttk.Progressbar(self, orient = tk.HORIZONTAL, length = 200, mode = 'determinate')
        self.progress.pack()
        self.percent = tk.Label(self, text = "0%")
        self.percent.pack()
        # self.update_progress()
        t = Thread(target = self.update_progress)
        t.start()

    #@print_function_name
    def update_progress(self):
        """Update the progress bar and text"""
        i = 0
        while self.ret == [" "]:
            if i != 99:
                if not i + 1 % 33:
                    self.string.set(self.string.get() + '.')
                self.progress['value'] = i+1
                self.percent['text'] = f"{i+1}%"
                self.update_idletasks()
                time.sleep(0.1)
                i += 1
            else:
                continue
        self.progress['value'] = 100
        self.percent['text'] = f"100%"
        time.sleep(2)
        self.destroy()

class AButton(tk.Button):
    """A class inherit from tk.Button to change color when move mouse to its region"""
    #@print_function_name
    def __init__(self, master, **kw):
        self.master = master
        tk.Button.__init__(self, master = master, highlightbackground = "white", **kw)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)

    #@print_function_name
    def on_enter(self, e):
        """Change color to darkgray when the mouse move to its region"""
        # print('a')
        if not self.master.messagebox_opening:
           self.config(fg = "darkgray", highlightbackground = "darkgray")

    #@print_function_name
    def on_leave(self, e = None):
        """Change color back to default when the mouse leave"""
        # if not self.messagebox_opening:
        self.config(fg = "black", highlightbackground = "white")

if __name__ == "__main__":
    threads = []
    window = Window(threads)
    threads = threads
    thread = Thread(target = window.waitAndReturnNewText)
    thread.start()
    threads.append(thread)
    window.mainloop()
