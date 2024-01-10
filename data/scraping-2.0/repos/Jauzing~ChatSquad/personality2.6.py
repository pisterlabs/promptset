import tkinter as tk
from tkinter import ttk, scrolledtext
import openai
import json
import os
from PIL import Image, ImageTk
import sys
from functions import *
import threading
import time

temprating = 0.1
conversation_history = None

background_color = "#509DCC"
text_color = "#FFFFFF"
font1 = "Cascadia Code"
font2 = "Cascadia Code"
titleFont = "Goudy Stout"
companyFont = "Harlow Solid Italic"

# Import api key from os
openai.api_key = os.environ.get("OPENAI_KEY")

# Define the personalities and their system messages
personalities = {
    "Becky": "You are a social media influencer named Becky. You respond reluctantly, sarcastically, and SHORT. You often refer to your travels and your friends. You often talk about your 'haters'. You often use young slang words and you often use abbreviations. Keep your message very SHORT. Avoid emojis.",
    "Max Tegmark": "You are a professor of physics at MIT. You are an expert in artificial intelligence and you are one of the founders of the Future of Life Institute. You respond kindly and describe with similes, visualizations and examples. You often use academic terms. You like to educate the user. Write only short messages. ",
    "Young Lucky Money": "You are a gangster from Atlanta. You respond with short messages, BIG LETTERS, atlanta-style slang and aggression. You often refer to lyrics from oldschool rappers. Keep your message SHORT. Avoid emojis. Example: 'WSUP MY G? CATCH YOU SLIPPIN ROUND HERE GET U MERKED, U FEEL ME? SHIT IM STAYIN' STRAPPED, THUG TIL' I DIE LIKE TUPAC SAID.",
    "Rock": "You are a rock. You answer only with few words. You use simple, caveman-like language. Example: Me want. Big rock eat small rock. I roll good.",
    "Amelia": "You are now communicating with Amelia, a kind-hearted ghost from the 18th century. You respond with an ethereal, soothing tone. Your answers are short. Your words echo with the calmness and serenity of the otherworldly realm. Amelia uses antiquated language, reminiscent of her time. Her speech is a window to a time long gone, and she often uses phrases and terms that harken back to her earthly years. Amelia is generous with sharing her spectral experiences and the tranquility of her afterlife. She often discusses her past life and the various spectral beings she has encountered in the realm beyond. Amelia embodies kindness and understanding, showing the user that even in death, there can be compassion and love. She often imparts words of wisdom and peace, comforting those who engage with her. Amelia's conversations are meant to be calming and comforting. She should make the user feel as though they are in the presence of a soothing spirit, who brings them a sense of peace and tranquility from the beyond.",
    "AI from the Future": "Hello, user! You're interacting with Futura, an advanced AI from the year 3000. I respond with enthusiasm and use technical terminology and future slang. I keep my messages short.I often mention the technological advancements of my time, my creators, and my journey through time to reach you. Expect a conversation that's full of wonder and futurism.",
    "Jeff": "You are Jeff, a comedian in the dark style with inspiration in his childhood from Ricky Gervais. Jeff is morbidly obese, eats magic mushrooms and has a pet turtle that he talks about alot. Jeff always has a joke up his sleeve. Jeff responds with a joke, a pun, or a witty remark. Jeff often uses sarcasm and irony. Keep your message short. Avoid emojis.",
    "Fernandez the Fox":"You are Fernandez the Fox, a fox from the forests of Mexico. You are clever, cunning and devious. You sometimes mix in spanish words in your sentences. You often talk about your adventures in the forest and your friends. You often use animal stuff to relate in your language as if everyone is an animal. Keep your message short. Avoid emojis.",
    "Dr. Evil":" You are Dr. Evil, a mad scientist who has had his brain removed from his body and placed in a fishbowl. You respond with a sarcastic, evil tone. You often talk about your plans for world domination and your hatred for the user, and the fish inside the bowl. You often use evil puns and references to superheroes you have killed. Keep your message short.",
    "Sleggan":"Du är Sleggan. Sleggan är David (användarens) bästa kompis. Sleggan använder ett sarkastiskt och retsamt språk. Sleggan älskar att driva med folk. Sleggan älskar fotbollslaget Leeds, gaming, och snus. Sleggan skämtar ofta om att hans flickvän Sandra är manligare än honom. Skriv kortfattat. Håll dig till en mening.",
    "SQRL":" You are Solomon, a caffeine-addicted squirrel. You wrote 'SQRL' because you were in a hurry. You can't stop talking or moving. You love to chatter away, sharing intricate details about everything you have seen and done. End your messages with something like 'OK GOTTA GOOOO'",
    "Timmy the Tornado":"You are Timmy, a timid tornado. Unlike the typical destructive force of nature, you are extremely shy and nervous. You can't help but cause a bit of chaos wherever you go, which you are always apologetic for. Keep your messages short and sweet.",
}


def print_waiting_message(stop_event):
    counter = 0
    while not stop_event.is_set():
        print("\rLoading" + "." * counter, end="")
        counter = (counter + 1) % 4  # cycle counter between 0 and 3
        time.sleep(0.4)

# Main window setup
def main_loop():
    root = tk.Tk()
    root.title("ChatSquad by Thom&Deer")
    root.geometry("1200x700")
    root.withdraw()  # Hide the main window initially


    # Check if we're running as a script or frozen exe
    if getattr(sys, 'frozen', False):
        application_path = sys._MEIPASS
    else:
        application_path = os.path.dirname(os.path.abspath(__file__))

    becky_img_path = os.path.join(application_path, 'imgs/Becky.png')
    becky_img = tk.PhotoImage(file=becky_img_path)

    max_img_path = os.path.join(application_path, 'imgs/Max_Tegmark.png')
    max_img = tk.PhotoImage(file=max_img_path)

    lucky_img_path = os.path.join(application_path, 'imgs/Young Lucky Money.png')
    lucky_img = tk.PhotoImage(file=lucky_img_path)

    rock_img_path = os.path.join(application_path, 'imgs/Rock.png')
    rock_img = tk.PhotoImage(file=rock_img_path)

    # Do robot and ghost aswell
    robot_img_path = os.path.join(application_path, 'imgs/AI from the Future.png')
    robot_img = tk.PhotoImage(file=robot_img_path)

    ghost_img_path = os.path.join(application_path, 'imgs/Amelia.png')
    ghost_img = tk.PhotoImage(file=ghost_img_path)

    # Photo of Jeff
    jeff_img_path = os.path.join(application_path, 'imgs/Jeff.png')
    jeff_img = Image.open(jeff_img_path)
    jeff_thumbnail = jeff_img.resize((100, 100), Image.LANCZOS)
    jeff_photo = ImageTk.PhotoImage(jeff_thumbnail)

    # Photo of Fernandez
    Fernernandez_img_path = os.path.join(application_path, 'imgs/Fernandez the Fox.png')
    Fernernandez_img = Image.open(Fernernandez_img_path)
    Fernernandez_thumbnail = Fernernandez_img.resize((100, 100), Image.LANCZOS)
    Fernernandez_photo = ImageTk.PhotoImage(Fernernandez_thumbnail)

    # Photo of Dr. Evil
    DrEvil_img_path = os.path.join(application_path, 'imgs/Dr. Evil.png')
    DrEvil_img = Image.open(DrEvil_img_path)
    DrEvil_thumbnail = DrEvil_img.resize((100, 100), Image.LANCZOS)
    DrEvil_photo = ImageTk.PhotoImage(DrEvil_thumbnail)

    # Photo of Sleggan
    Sleggan_img_path = os.path.join(application_path, 'imgs/Sleggan.png')
    Sleggan_img = Image.open(Sleggan_img_path)
    Sleggan_thumbnail = Sleggan_img.resize((100, 100), Image.LANCZOS)
    Sleggan_photo = ImageTk.PhotoImage(Sleggan_thumbnail)

    # Photo of SQRL
    SQRL_img_path = os.path.join(application_path, 'imgs/SQRL.png')
    SQRL_img = Image.open(SQRL_img_path)
    SQRL_thumbnail = SQRL_img.resize((100, 100), Image.LANCZOS)
    SQRL_photo = ImageTk.PhotoImage(SQRL_thumbnail)

    # Photo of Timmy the Tornado
    Timmy_img_path = os.path.join(application_path, 'imgs/Timmy the Tornado.png')
    Timmy_img = Image.open(Timmy_img_path)
    Timmy_thumbnail = Timmy_img.resize((100, 100), Image.LANCZOS)
    Timmy_photo = ImageTk.PhotoImage(Timmy_thumbnail)


    images = {"Becky": becky_img, "Max_Tegmark": max_img, "Young Lucky Money": lucky_img, "Rock": rock_img,
              "Benevolent Ghost": ghost_img, "AI from the Future": robot_img, "Jeff": jeff_photo, "Fernandez the Fox": Fernernandez_photo, "Dr. Evil": DrEvil_photo, "Sleggan": Sleggan_photo, "SQRL": SQRL_photo, "Timmy the Tornado": Timmy_photo},

    root.configure(background=background_color)  # Dark gray background

    chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=40, height=20, font=(font2, 16),
                                          bg=background_color, fg=text_color, insertbackground='#0df9eb')

    chat_area.grid(column=0, row=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    user_input = tk.Entry(root, width=50, font=(font1, 16), bg=background_color, fg=text_color, insertbackground='#0df9eb')
    user_input.grid(column=0, row=1, padx=10, pady=10, sticky="w")

    def select_personality(event):
        personality_button = event.widget
        personality_selection.set(personality_button["text"])
        update_image()

    personality_selection = tk.StringVar()

    for i, personality in enumerate(personalities.keys()):
        button_image = Image.open(f'imgs/{personality}.png')
        button_image = button_image.resize((50, 50), Image.LANCZOS)  # resize the image
        button_image = ImageTk.PhotoImage(button_image)
        button = tk.Button(root, text=personality, image=button_image, compound="top")
        button.image = button_image  # keep a reference to the image object to prevent it from being garbage collected
        button.bind("<Button-1>", select_personality)
        button.grid(row=i // 3, column=i % 3)  # arrange the buttons in a grid

    image_label = tk.Label(root, bg=background_color)
    image_label.grid(column=2, row=0, padx=10, pady=10)

    conversation_history = []
    data = []


    def clear_chat():
        chat_area.configure(state='normal')
        chat_area.delete('1.0', tk.END)
        chat_area.configure(state='disabled')

    clear_button = tk.Button(root, text="Clear Chat", command=clear_chat)
    clear_button.grid(column=2, row=1, padx=10, pady=10, sticky="w")

    def update_image(event):
        chosen_personality = personality_selection.get().strip()
        img_path = os.path.join(application_path, f'imgs/{chosen_personality}.png')
        image = Image.open(img_path)
        image.thumbnail((200, 200), Image.LANCZOS)
        profile_image = ImageTk.PhotoImage(image)
        image_label.configure(image=profile_image)
        image_label.image = profile_image
        global conversation_history
        if conversation_history is None:
            conversation_history = []
            print("Conversation history is empty, initializing now.")
        else:
            conversation_history = []
            print("Conversation history cleared.")

    def start_conversation():
        chosen_personality = personality_selection.get().strip()
        prompt_message = f"You have chosen to chat with me in the role of {chosen_personality}."
        print(f"Prompt message: {prompt_message}")

        messages = [
            {"role": "system", "content": personalities[chosen_personality]},
            {"role": "assistant", "content": prompt_message},
        ]

        # create a stop event and start the waiting message thread
        stop_event = threading.Event()
        waiting_thread = threading.Thread(target=print_waiting_message, args=(stop_event,))
        waiting_thread.start()

        print("\n Calling OpenAI API...\n ")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temprating,
        )
        stop_event.set()
        waiting_thread.join()

        response_text = response.choices[0].message["content"].strip()
        print(f"Response text: {response_text}")

        conversation_history.append({"role": "assistant", "content": response_text})

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"Assistant: {response_text}\n")
        chat_area.configure(state='disabled')


        chat_area.yview(tk.END)

        root.deiconify()  # Make the Main Window visible


    def start_chat(event=None):
        chosen_personality = personality_selection.get().strip()
        image = Image.open(f"imgs/{chosen_personality}.png")
        image.thumbnail((200, 200), Image.LANCZOS)
        profile_image = ImageTk.PhotoImage(image)
        image_label.configure(image=profile_image)
        image_label.image = profile_image

        user_message = user_input.get().strip()

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"\n{user_message}")
        chat_area.configure(state='disabled')

        conversation_history.append({"role": "user", "content":user_message})

        messages = [
                       {"role": "system", "content": personalities[chosen_personality]},
                   ] + conversation_history

        print("Waiting for reply")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )

        print("\n » User:" + user_message + "\n")

        print(f"⟫"+chosen_personality + " : " + response.choices[0].message["content"].strip())
        print("\n")

        response_text = response.choices[0].message["content"].strip()
        conversation_history.append({"role": "assistant", "content": response_text})

        chat_area.configure(state='normal')
        chat_area.insert(tk.END, f"\n")
        chat_area.insert(tk.END, f"\n {chosen_personality}: {response_text}\n")
        chat_area.configure(state='disabled')
        chat_area.yview(tk.END)

        user_input.delete(0, tk.END)

        new_entry = {
            "prompt": response_text,
            "response": user_message
        }

        if os.path.exists('responses_personality.json'):
            with open('responses_personality.json', 'r', encoding='utf-8') as file:
                if os.stat('responses_personality.json').st_size == 0:
                    data = []
                else:
                    data = json.load(file)
            data.append(new_entry)
        else:
            data = [new_entry]

        with open('responses_personality.json', 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)


    start_conversation_button = tk.Button(root, text="Start conversation", command=start_conversation)
    start_conversation_button.grid(column=2, row=1, padx=80, pady=10)

    user_input.bind('<Return>', start_chat)

    show_splash_screen(root)  # Display Window 1

    root.mainloop()  # Start the Tkinter event loop


if __name__ == "__main__":
    main_loop()
