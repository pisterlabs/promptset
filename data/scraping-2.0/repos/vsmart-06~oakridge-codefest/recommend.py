import tkinter as tk
from tkinter import ttk
from sidebar import Sidebar
import openai
import os
import dotenv
import textwrap

dotenv.load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

class Recommendation:
    def __init__(self, username: str):
        self.username = username
        self.root = tk.Tk()
        self.root.title("Recommendations")
        self.root.geometry("600x400")
        self.root.tk.call("source", "./oakridge-codefest/forest-dark.tcl")
        ttk.Style().theme_use("forest-dark")

        self.frame = ttk.Frame(self.root)
        self.frame.grid(row = 0, column = 1, padx = 10)

        self.warning = ttk.Label(self.frame, text = "Need some recommendations on events you can possibly hold? \nTry clicking the AI recommendation button. \nBeware it is still experimental and will take some time!!")
        self.warning.grid(row = 0, column = 0, pady = 10)
        button = ttk.Button(self.frame, text = "Get Recommendations", style = "Accent.TButton", command = self.giveReccomendation)
        button.grid(row = 1, column = 0)

        self.root.update()
        sidebar = Sidebar(self.root, self.username)

        self.root.mainloop()
    
    def giveReccomendation(self):
        text = self.giverec()
        try:
            self.myLabel.destroy()
        except:
            pass
        self.myLabel = ttk.Label(self.frame, text = text)
        self.myLabel.grid(row = 2, column = 0, pady = 10)

    def giverec(self):
        openai.api_key = API_KEY
        file1 = open("./oakridge-codefest/text1.txt","r").read()
        file2 = open("./oakridge-codefest/text2.txt","r").read()
        file3 = open("./oakridge-codefest/text3.txt","r").read()
        file4 = open("./oakridge-codefest/text4.txt","r").read()
        file5 = open("./oakridge-codefest/text5.txt","r").read()
        dataset = [file1, file2, file3, file4, file5]

        model = "davinci"
        prompt = "Fine-tune the model to give shorter but similar results and give only 1 reccomendation on possible events to be conducted to be more environmentally conscious: " + " | " .join(dataset)   


        response = openai.Completion.create(
            engine = model,
            prompt = prompt,
            temperature = 0.5,
            max_tokens = 1024,
            top_p = 1,
            frequency_penalty = 1,
            presence_penalty = 1,
        )

        recc = response["choices"][0]["text"]
        recc = textwrap.fill(recc, width = 50)
        recc = recc.replace("|", "\n")
        return recc