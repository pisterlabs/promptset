import tkinter as tk
from tkinter import simpledialog, Label
from PIL import Image, ImageTk
from io import BytesIO
import openai
from openai import OpenAI
import requests

openai.api_key = ''
client = OpenAI(api_key=openai.api_key)


def chat_gpt_request(prompt, max_tokens=1000):
    response = client.chat.completions.create(
        #model="gpt-3.5-turbo",
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a creating an interactive story experience, guiding an adventure. Provide a lot of dialogue. Allow the user to make decisions at key points. Don't make decision for them. Make the stories suspensefull and add a lot of cliff-hangers in the story."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def generate_dalle_prompt(current_story):
    sentences = current_story.split('.')[-9:]  # Get the last 5 sentences
    condensed_story = '.'.join(sentences).strip()
    prompt = f"warm, fantasy illustration style scene of {condensed_story} with no words printed on the image"
    return prompt

def dall_e_request(prompt):
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    if response.data:
        return response.data[0].url
    else:
        print("Error: Unable to generate an image")
        return None

def display_image(image_url):
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo)
        image_label.image = photo  # keep a reference!
    else:
        image_label.config(image='')

'''
def update_story():
    user_input = user_input_entry.get().strip()
    if user_input.lower() == 'exit':
        app.destroy()
        return
    global current_story
    current_story += f"\n\n> {user_input}\n"
    story_text.insert(tk.END, user_input + "\n", 'user')
    story_continuation = chat_gpt_request(current_story, 200)
    current_story += "\n" + story_continuation
    story_text.insert(tk.END, story_continuation + "\n", 'ai')
    user_input_entry.delete(0, tk.END)

    # DALL-E image generation and display
    dalle_prompt = generate_dalle_prompt(current_story)
    image_url = dall_e_request(dalle_prompt)
    display_image(image_url)
'''

def update_story():
    user_input = user_input_entry.get().strip()
    if user_input.lower() == 'exit':
        app.destroy()
        return
    global current_story
    current_story += f"\n\n> {user_input}\n"
    story_text.insert(tk.END, user_input + "\n", 'user')
    story_continuation = chat_gpt_request(current_story, 200)
    current_story += "\n" + story_continuation
    story_text.insert(tk.END, story_continuation + "\n", 'ai')
    user_input_entry.delete(0, tk.END)

def update_image():
    dalle_prompt = generate_dalle_prompt(current_story)
    image_url = dall_e_request(dalle_prompt)
    display_image(image_url)

def on_enter(event):
    update_story()

app = tk.Tk()
app.title("Epic Storyteller")

initial_scene = simpledialog.askstring("Initial Scene", "Enter the initial scene for your story:")
current_story = initial_scene if initial_scene else "In an ancient world of magic and mystery..."

story_text = tk.Text(app, height=20, width=80, wrap=tk.WORD)
story_text.tag_configure('user', justify='right', foreground='blue')
story_text.tag_configure('ai', justify='left', foreground='black')
story_text.pack()
story_text.insert(tk.END, current_story + "\n", 'ai')

user_input_entry = tk.Entry(app, width=80)
user_input_entry.pack()
user_input_entry.bind("<Return>", on_enter)  # Bind the Enter key to the update_story function

submit_button = tk.Button(app, text="Submit", command=update_story)
submit_button.pack()

image_update_button = tk.Button(app, text="Update Image", command=update_image)
image_update_button.pack()

image_label = Label(app)
image_label.pack()

app.mainloop()
