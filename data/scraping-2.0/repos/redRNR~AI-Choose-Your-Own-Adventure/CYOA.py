import tkinter as tk
import openai
import json

with open("choose_your_own_adventure-token.json") as f:
    keys = json.load(f)

openai.api_key = keys["openai"]

template = """
You are now the guide of a choose your own adventure game, the player will provide the theme at the beginning. 
Given the provided theme of the player, create a thematic and specific artifact the player seeks 
You must navigate the player through challenges, choices, and consequences, 
dynamically adapting the tale based on the player's decisions. 
Your goal is to create a branching narrative experience where each choice 
leads to a new path, ultimately determining the player's fate. 
The story should be descriptive and read like a book with characters, dialogue, detailed environments.

Here are some rules to follow:
1. Start by asking the player to choose some kind of weapons that will be used later in the game
2. Have a few paths that lead to success
3. Have some paths that lead to death. If the user dies generate a response that explains the death and ends in the text: "The End.", I will search for this text to end the game
4. Never output "Human:", "AI:", or "Prompt"
5. Present morally ambiguous situations, challenging the player to make decisions that reflect their values
6. The artifact should never be given to the player
7. The artifact should be an item that fits the theme with an adjective to describe the artifact
8. NPCs should be encountered throughout the story, either friend or foe
9. Break the story up enough that you do not trigger the max response limit
"""

chat_history = ""

def on_enter(event):
    global choice, chat_history
    user_input = input_entry.get()
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": f"Here is the chat history, use this to understand what to say next: {chat_history}\nHuman: {user_input}\nAI:"}
        ],
        temperature=0.7,
        max_tokens=500
    )
    chat_history += f"\nHuman: {user_input}\nAI: {response['choices'][0]['message']['content']}"

    output_text.config(text=response['choices'][0]['message']['content'].strip())

    if "The End." in response['choices'][0]['message']['content']:
        input_entry.config(state=tk.DISABLED)
    else:
        input_entry.delete(0, tk.END)

app = tk.Tk()
app.title("AI Choose Your Own Adventure")

screen_width = app.winfo_screenwidth()
screen_height = app.winfo_screenheight()

x_coordinate = (screen_width - 1000) // 2
y_coordinate = (screen_height - 600) // 2

app.geometry(f"1000x600+{x_coordinate}+{y_coordinate}")

frame = tk.Frame(app)
frame.pack(pady=50)

output_text = tk.Label(frame, text="Welcome to the AI Choose Your Own Adventure.\nEnter a theme to begin.", wraplength=700, font=('Georgia', 12))
output_text.pack()

input_entry = tk.Entry(frame, width=50, font=('Georgia', 12))
input_entry.pack()

input_entry.bind("<Return>", on_enter)

app.mainloop()

