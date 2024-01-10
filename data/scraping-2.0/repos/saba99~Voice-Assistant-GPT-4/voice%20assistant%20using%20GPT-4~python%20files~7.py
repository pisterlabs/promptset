from tkinter import *
import openai

openai.api_key = 'Your_API_KEY'

window = Tk()
window.geometry("600x600")
window.title("AI Chat App")

lbl = Label(window, text="How can I help you?").pack()

ent = Entry(window)
ent.pack()

txt = Text(window, height = 25, width = 50)
txt.pack()

conversation = ""

def ask(event):
    global conversation
    query = ent.get()
    ent.delete(0,END)
    conversation = conversation + "\n\n" + query
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are an advisor"},
        {"role": "assistant", "content": conversation},
        {"role": "user", "content": query}
    ]
    )
    answer = response["choices"][0]["message"]["content"]
    txt.insert(END, answer + "\n\n")

btn = Button(window, text="submit", command=ask)
btn.pack()

btn_exit = Button(window, text ="Exit", command=window.destroy).pack()

window.bind("<Return>", ask)

window.mainloop()
