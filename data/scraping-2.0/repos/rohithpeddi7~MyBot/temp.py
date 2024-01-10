from tkinter import *
from textwrap3 import wrap

# GUI
root = Tk()
root.title("MyBot")

BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

import os
import openai

openai.api_key = "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"
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
	return response["choices"][0]["text"].strip()

# Send function
def send():
	send = "You : " + e.get()
	txt.insert(END, "\n" + send)
	user = e.get().lower()
	answer = gpt(user)
	txt.insert(END, "\n")
	for ans in wrap("Bot : "+answer,width=65):
		txt.insert(END, ans+"\n" )
	e.delete(0, END)


lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="AI assistant", font=FONT_BOLD, pady=10, width=20, height=1).grid(
	row=0)

txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)

scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)

e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=50)
e.grid(row=2, column=0)

send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			command=send).grid(row=2, column=1)

root.mainloop()