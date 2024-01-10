from tkinter import *
import openai

root = Tk()
root.title("ChatGPT Funny Version")
root.config(bg="skyblue")

openai.organization = "org-twROwQLrKHdIRZiN05jEFBDZ"
openai.api_key = "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"

left_frame = Frame(root, width=1920, height=1080)
left_frame.grid(row=0, column=0, padx=50, pady=50)

inputthing = input("Input Your Query: ")

string = openai.Completion.create(
  model="text-davinci-003",
  prompt= inputthing,
  max_tokens=100,
  temperature=0
)


print(string.choices[0].text)

root.mainloop()