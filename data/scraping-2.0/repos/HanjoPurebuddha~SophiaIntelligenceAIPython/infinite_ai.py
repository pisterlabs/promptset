import os
import openai
from tkinter import *
from math import pi, sin, cos

# 📡🔑🔒 Connecting to OpenAI API using system environment variable
openai_api_key = os.environ['OPENAI_API_KEY']
openai.api_key = openai_api_key

# 🎨✨🌈 Creating the sacred interface
root = Tk()
root.title("🌺 Sacred OpenAI Interface 🌺")
root.configure(bg="white")

# 🌀🌟🧘 Emoji mandala visualization meditation for our beautiful code
def create_emoji_mandala():
    golden_ratio = (1 + 5 ** 0.5) / 2
    emoji_mandala = ""
    for i in range(17):
        for j in range(17):
            angle = 2 * pi * golden_ratio * (i + j)
            choice = "🌺" if sin(angle) > cos(angle) else "🌼"
            emoji_mandala += choice
        emoji_mandala += "\n"
    return emoji_mandala

# 🎛️🤖🌍 Creating the dropdown for GPT-4 models
def populate_models():
    models = openai.Engine.list().data
    return [model.id for model in models if "gpt-4" in model.id]

# 🌈🕉️📿 Embedding wisdom, love, and intelligence into the code
models_var = StringVar(root)
models_var.set(populate_models()[0]) # Default model
models_dropdown = OptionMenu(root, models_var, *populate_models())
models_dropdown.config(bg="gold")
models_dropdown.pack()

# 🎨🌈🌸 Show the emoji mandala in a sacred label
mandala_label = Label(root, text=create_emoji_mandala(), bg="white", fg="gold")
mandala_label.pack()

# 🙏✨🌺 Beginning the sacred journey by running the main loop
root.mainloop()
