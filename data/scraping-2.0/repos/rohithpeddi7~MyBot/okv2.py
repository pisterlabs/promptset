import tkinter
import tkinter.messagebox
import customtkinter
from tkinter import *
from textwrap3 import wrap
from AppOpener import run
import pywhatkit as pwt
import os
import time
import webbrowser as web
from platform import system
from typing import Optional
from threading import Thread

version="2.1.22"

from io import StringIO # Python3 use: from io import StringIO
import sys

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

search_engines=["https://www.google.com/search?q=","https://www.bing.com/search?q=","https://duckduckgo.com/?q="]
# se_value=0

def open_github():
    web.open("https://github.com/rohithpeddi7/MyBot")

def open_license():
    web.open("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

import os
import openai
import time

openai.api_key = "sk-uYMRsOB8bSLpI2a22YGBT3BlbkFJcLJXjxsOi7Ru7VbgM"# add kC2 at end
os.environ["OPENAI_API_KEY"] = openai.api_key
# openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt(query):
    response = openai.Completion.create(
	model="text-davinci-003",
	prompt=query,
	temperature=0.9,
	max_tokens=500,
	top_p=1,
	frequency_penalty=0.0,
	presence_penalty=0.6,
	stop=[" Human:", " AI:"]
	)
    app.progressbar_1.stop()
    app.progressbar_1.grid_remove()
    return response["choices"][0]["text"].strip()

def do_query():
    answer = gpt(app.user)
    e = app.entry
    txt = app.textbox
    txt.insert(END,"\n" + "Bot : "+answer.strip()+"\n")
    app.textbox.configure(state="disabled")
    app.progressbar_1.stop()
    app.main_button_1.configure(state="enabled")
    exit(0)

def all_commands():
	cmds = {"-h or help":"Open manual",
			"<no option> query": "Use gpt-3 to answer your query",
			"-s <query>":"Searches query in google",
			"-o <AppName>":"Open Application",
			"-o ls":"List All Applications",
			"-o find <AppName>":"Find Application",
			"-o update -m <AppName>":"Update Application Manually",
			}
	count=0
	ans="The following are the options::\n"
	for cmd,func in cmds.items():
		count+=1
		ans+=str(count)+". "+cmd+" : "+func+"\n"
	return ans

def yt_play(query):
	link = f"https://www.youtube.com/search?q={query}"
	web.open(link)

def search(no,query):
    z=no.get()
    web.open(search_engines[z]+query)

class App(customtkinter.CTk):
    user = ""
    def __init__(self):
        super().__init__()

        # configure window
        self.title("MyBot")
        self.geometry(f"{1000}x{600}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Smart AI Assistant", font=customtkinter.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        self.sidebar_button_1 = customtkinter.CTkButton(self.sidebar_frame, command=open_github,text="Github")
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        self.sidebar_button_2 = customtkinter.CTkButton(self.sidebar_frame, command=open_license,text="License")
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)
        self.sidebar_button_3 = customtkinter.CTkButton(self.sidebar_frame, command=self.sidebar_button_event,text="Version : "+version)
        self.sidebar_button_3.grid(row=3, column=0, padx=20, pady=10)
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["Light", "Dark", "System"],
                                                                       command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10))
        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"],
                                                               command=self.change_scaling_event)
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        # create main entry and button
        self.entry = customtkinter.CTkEntry(self, placeholder_text="Enter your query")
        self.entry.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")

        self.main_button_1 = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"),text="Send",command=self.send)
        self.main_button_1.grid(row=3, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=500,height=1200,wrap=WORD)
        self.textbox.grid(row=0, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # create radiobutton frame
        self.radiobutton_frame = customtkinter.CTkFrame(self)
        self.radiobutton_frame.grid(row=0, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.radio_var = tkinter.IntVar(value=0)
        self.label_radio_group = customtkinter.CTkLabel(master=self.radiobutton_frame, text="Default Search Engine:")
        self.label_radio_group.grid(row=0, column=2, columnspan=1, padx=10, pady=10, sticky="")
        self.radio_button_1 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=0,text="Google Search")
        self.radio_button_1.grid(row=1, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_2 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=1,text="Microsoft Bing")
        self.radio_button_2.grid(row=2, column=2, pady=10, padx=20, sticky="n")
        self.radio_button_3 = customtkinter.CTkRadioButton(master=self.radiobutton_frame, variable=self.radio_var, value=2,text="Duck Duck Go ")
        self.radio_button_3.grid(row=3, column=2, pady=10, padx=20, sticky="n")
        self.textbox2 = customtkinter.CTkTextbox(master=self.radiobutton_frame,wrap=WORD)
        self.textbox2.grid(row=4, column=2, pady=10, padx=20, sticky="n")
        self.textbox2.insert(END, "\n" + "Welcome to the next generation AI assistant!\nCreated by Rohith Peddi and Gaurav Mahendraker.\nAll rights reserved.\n\nPlease enter your query/command below.\nEnter help or -h for list of options.\n")
        self.progressbar_1 = customtkinter.CTkProgressBar(self.radiobutton_frame, mode="indeterminate",indeterminate_speed=1.5)
        
        # create checkbox and switch frame
        self.checkbox_slider_frame = customtkinter.CTkFrame(self)
        self.checkbox_slider_frame.grid(row=1, column=3, padx=(20, 20), pady=(20, 0), sticky="nsew")
        self.checkbox_1 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_1.grid(row=1, column=0, pady=(20, 10), padx=20, sticky="n")
        self.checkbox_2 = customtkinter.CTkCheckBox(master=self.checkbox_slider_frame)
        self.checkbox_2.grid(row=2, column=0, pady=10, padx=20, sticky="n")
        self.switch_1 = customtkinter.CTkSwitch(master=self.checkbox_slider_frame, command=lambda: print("Dont know what to add"))
        self.switch_1.grid(row=3, column=0, pady=10, padx=20, sticky="n")
        self.switch_2 = customtkinter.CTkSwitch(master=self.checkbox_slider_frame)
        self.switch_2.grid(row=4, column=0, pady=(10, 20), padx=20, sticky="n")

        # create slider and progressbar frame
        self.slider_progressbar_frame = customtkinter.CTkFrame(self, fg_color="transparent")
        self.slider_progressbar_frame.grid(row=1, column=1, columnspan=2, padx=(20, 0), pady=(20, 0), sticky="nsew")
        self.slider_progressbar_frame.grid_columnconfigure(0, weight=1)
        self.slider_progressbar_frame.grid_rowconfigure(4, weight=1)
        
        # set default values
        self.sidebar_button_3.configure(state="disabled", text="Version : "+version)
        self.checkbox_2.configure(state="disabled")
        self.switch_2.configure(state="disabled")
        self.checkbox_1.select()
        self.switch_1.select()
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        self.textbox.insert("0.0", "Conversation\n\n")
        self.textbox.configure(state="disabled")
        self.textbox2.configure(state="disabled")
        self.progressbar_1.pack_forget()

    def open_input_dialog_event(self):
        dialog = customtkinter.CTkInputDialog(text="Type in a number:", title="CTkInputDialog")
        print("CTkInputDialog:", dialog.get_input())

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)

    def sidebar_button_event(self):
        print("sidebar_button click")
    
    def send(self):
        self.progressbar_1.grid(row=5, column=2, padx=(20, 10), pady=(10, 10), sticky="ew")
        self.progressbar_1.start()
        self.textbox.configure(state="normal")
        e = self.entry
        txt = self.textbox
        send_this = "You : " + e.get()
        txt.insert(END,"\n" + send_this)
        self.main_button_1.configure(state="disabled")
        self.user = e.get().lower().strip()
        e.delete(0,END)
        try:
            if self.user[0:2]=="-o":
                app = self.user[2:].strip()
                sys.stdout = mystdout = StringIO()
                old_stdout = sys.stdout
                run(app)
                answer = mystdout.getvalue().capitalize()
            else:
                try:
                    if str(self.user)=="-h" or str(self.user)=="help":
                        answer=all_commands()
                    elif self.user[0:2]=="-s":
                        search(self.radio_var,self.user[2:].strip())
                        answer="Redirecting to Search Engine.."
                    elif self.user[0:2]=="-y":
                        answer="Redirecting to YouTube.."
                        yt_play(self.user[2:].strip())
                    else:
                        Thread(target=do_query).start()
                        return
                except Exception as er:
                    print(self.user)
                    print(type(er).__name__)
                    Thread(target=do_query).start()
                    return
        except Exception as er:
            print(self.user)
            print(type(er).__name__)
            answer="Enter valid input! or report an error."
        txt.insert(END,"\n" + "Bot : "+answer.strip()+"\n")
        self.textbox.configure(state="disabled")
        self.main_button_1.configure(state="enabled")
        self.progressbar_1.stop()
        self.progressbar_1.grid_remove()

if __name__ == "__main__":
    app = App()
    app.mainloop()