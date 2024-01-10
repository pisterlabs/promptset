from openai import OpenAI
apifile = open('API_KEY.txt', 'r')
client = OpenAI(api_key=apifile.read().strip())

# Function to generate an image using DALL-E
def generate_image(prompt):
    """
    Generates an image with a prompt that is given as a parameter.
    """
    response = client.images.generate(model="dall-e-3",
                                        prompt=prompt,
                                        size="1024x1024",
                                        quality="standard",
                                        n=1)
    image_url = response.data[0].url
    return image_url


# The Tkinter application
import tkinter as tk
from tkinter import ttk  
from ttkthemes import ThemedTk
import io
import requests
from PIL import Image, ImageTk

# Function to handle the image prompt input
def handle_image_prompt():
    prompt_text = prompt_entered.get()
    image_url = generate_image(prompt_text)
    
    # Download the image from the URL
    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)
    
    # Open the image
    img = Image.open(image_bytes)
    img.thumbnail((800, 600)) 
    photo = ImageTk.PhotoImage(img)
    
    # Update the image_label with the new image
    image_label.configure(image=photo)
    image_label.image = photo


# Main window
window = ThemedTk(theme="Breeze")
window.title("DollEApp by George")
window.geometry("1080x1920")
window.configure(bg='#323232')

# Frame for the image placeholder
image_frame = ttk.Frame(window, width=800, height=600, relief="sunken", style='Breeze.TFrame')
image_frame.pack(side=tk.TOP, pady=20)
image_frame.pack_propagate(False)
# Image will appear here
image_label = ttk.Label(image_frame, text="Image will appear here", style='Breeze.TLabel') 
image_label.pack(expand=True)

# Frame for prompt input and button
input_frame = ttk.Frame(window, padding=10, style='Breeze.TFrame')   
input_frame.pack(side=tk.TOP, pady=(10, 0), fill='x')   
# Prompt Input
prompt_entered = ttk.Entry(input_frame, width=100)
prompt_entered.pack(side=tk.TOP, padx=10, expand=True, fill='x') # Changed side to TOP
# Button - Now appears below the input field
submit_button = ttk.Button(input_frame, text="Load Image", command=handle_image_prompt)
submit_button.pack(side=tk.TOP, padx=10, pady=(5, 0)) # Changed side to TOP and added vertical padding

# Start the tkinter loop
window.mainloop()