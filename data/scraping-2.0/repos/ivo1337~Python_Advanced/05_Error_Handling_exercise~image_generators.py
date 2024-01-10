import openai
import tkinter as tk

openai.api_key = ''


def get_image_url():
    response = openai.Image.create(
        prompt=input_field.get(),
        n=1,
        size="256x256"
    )

    image_url = response['data'][0]['url']

    return image_url


def render_image():
    image_url = get_image_url()


window = tk.Tk()
window.title("Image generator")
window.geometry("600x600")

input_field = tk.Entry(window)
input_field.place(x=125, y=125)

button = tk.Button(window, text="Create", height=1, command=render_image())
button.place(x=260, y=120)

window.mainloop()
