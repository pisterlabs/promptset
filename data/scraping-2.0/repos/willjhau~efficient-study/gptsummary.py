import openai as oi
from tkinter import *
from tkinter import filedialog

# Class to enable all things to do with summarising text
class Summariser():
    # Set the API key
    def __init__(self, text, API_KEY = 'YOUR KEY HERE'):
        oi.api_key = API_KEY
        self.text = text
        self.summary = None
    
    def summarise(self):
        # Set the options for and call the API
        response = oi.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a helpful research assistant."},
                {"role": "user", "content": f"Give me an concise analysis of this text, reducing its word count: {self.text}"}
            ],
        )
        self.summary = response["choices"][0]["message"]["content"]

    # Initialise the save file routine
    def saveFile(self):
        file = filedialog.asksaveasfile(defaultextension=".txt",
                                    filetypes = [
                                        ("Text file", ".txt"),
                                        ("HTML file", ".html"),
                                        ("All files", ".*"),
                                    ])
        file.write(self.summary)
        file.close()
    
    # Smoothes over any grammar mistakes
    def fixGrammar(self):
        response = oi.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are a grammar correcter."},
                {"role": "user", "content": f"Please correct the grammar mistakes in the following passage without losing any meaning: {self.text}"}
            ],
        )
        self.summary = response["choices"][0]["message"]["content"]

    # Set different text
    def setText(self, text):
        self.text = text

    # Get the saved summary attribute
    def getSummary(self):
        return self.summary













# def callAPI(text):

#     response = oi.ChatCompletion.create(
#         model = "gpt-3.5-turbo",
#         messages = [
#             {"role": "system", "content": "You are a helpful research assistant."},
#             {"role": "user", "content": f"Give me an consise analysis of this text: {text}"}
#         ],
#     )
#     summary = response["choices"][0]["message"]["content"]
#     print(summary)
#     return summary

# def save_file():
#     file = filedialog.asksaveasfile(defaultextension=".txt",
#                                     filetypes = [
#                                         ("Text file", ".txt"),
#                                         ("HTML file", ".html"),
#                                         ("All files", ".*"),
#                                     ])
#     file_text = str(text.get(1.0, END))
#     file.write(file_text)
#     file.close()

# def main(text):
#     summary = callAPI(text)




"""
Below defines the text box and save button
"""


#     window = Tk()
#     button = Button(text="save", command=save_file)
#     button.pack(side="bottom")


#     text = Text(window, wrap = "word")
#     text.insert("1.0", summary)
#     text.pack()
#     window.mainloop()
