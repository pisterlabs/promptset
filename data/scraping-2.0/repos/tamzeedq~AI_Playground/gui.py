import tkinter as tk
from PIL import ImageTk
from openai_api import generate_image

# Create the main window
app = tk.Tk()
app.geometry("500x500")
app.title("Playground")

# Create a prompt box
prompt_box = tk.Entry(app)
prompt_box.place(x=50, y=0, width=400, height=40)

# Function to generate an image
def call_dalle():
    generate_image(prompt_box.get())
    pass

# Create a button
generate_button = tk.Button(app, text="Generate", command=call_dalle)
generate_button.place(x=200, y=50, width=100, height=40)

# Start the main event loop
app.mainloop()
