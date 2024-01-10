from tkinter import *
import openai
from gtts import gTTS
import os

openai.api_key = 'Your_API_KEY'

window = Tk()
window.geometry("600x600")
window.title("Story Time App")

story = "Tell me a story about"

lbl = Label(window, text="What story would you like to hear?").pack()

ent = Entry(window)
ent.pack()

txt = Text(window, height=25, width=60)
txt.pack()

def ask(event):
    global story
    query = ent.get()
    ent.delete(0, END)
    story = story + "\n\n" + query
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "you are a story teller"},
            {"role": "assistant", "content": story},
            {"role": "user", "content": query}
        ]
    )
    answer = response["choices"][0]["message"]["content"]
    txt.insert(END, answer + "\n\n")
    speech = gTTS(answer)
    speech.save('speech.mp3')
    os.system("mpg123 speech.mp3") #MacOS = afplay, Ubuntu = mpg123
    os.system("rm speech.mp3")

btn = Button(window, text="submit", command=ask)
btn.pack()

btn_exit = Button(window, text="Exit", command=window.destroy).pack()

window.bind("<Return>", ask)

window.mainloop()
