import random
import os
import tkinter as tk
from tkinter import messagebox
import openai

# Set the openai.api_key to the value of the OPENAI_API_KEY environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

class TarotCard:
    def __init__(self, name):
        self.name = name

# Install the openai module
os.system('pip install openai')

class TarotDeck:
    def __init__(self):
        self.cards = []
        self.load_cards()

    def load_cards(self):
        major_arcana = [
            "The Fool",
            "The Magician",
            "The High Priestess",
            "The Empress",
            "The Emperor",
            "The Hierophant",
            "The Lovers",
            "The Chariot",
            "Strength",
            "The Hermit",
            "Wheel of Fortune",
            "Justice",
            "The Hanged Man",
            "Death",
            "Temperance",
            "The Devil",
            "The Tower",
            "The Star",
            "The Moon",
            "The Sun",
            "Judgement",
            "The World",
        ]

        for card_name in major_arcana:
            self.cards.append(TarotCard(card_name))

    def draw_card(self):
        return random.choice(self.cards)

class TarotReader:
    def __init__(self):
        self.deck = TarotDeck()

    def draw_cards(self, count=1):
        cards = []
        for _ in range(count):
            card = self.deck.draw_card()
            cards.append(card)
            self.deck.cards.remove(card)
        return cards

    def interpret_cards(self, cards):
        interpretations = []

        for card in cards:
            upright = random.choice([True, False])
            orientation = "Upright" if upright else "Reversed"
            prompt = f"Provide the {orientation.lower()} meaning of the {card.name} tarot card."

            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )

            meaning = response.choices[0].text.strip()
            interpretations.append(f"{card.name} ({orientation}): {meaning}")

        return interpretations

def gui_main():
    def draw_and_interpret():
        card_count = int(card_count_var.get())
        reader = TarotReader()
        cards = reader.draw_cards(card_count)
        interpretations = reader.interpret_cards(cards)

        interpretation_text = "\nYour Tarot reading:\n"
        for interpretation in interpretations:
            interpretation_text += f"{interpretation}\n"

        messagebox.showinfo("Tarot Reading", interpretation_text)

    root = tk.Tk()
    root.title("AI Tarot Reader based on ChatGPT")

    tk.Label(root, text="Welcome to the AI Tarot Reader based on ChatGPT!").pack()
    tk.Label(root, text="Please choose a number of cards to draw (1-3):").pack()

    card_count_var = tk.StringVar(root)
    card_count_var.set("1")

    tk.OptionMenu(root, card_count_var, "1", "2", "3").pack()

    draw_button = tk.Button(root, text="Draw Cards", command=draw_and_interpret)
    draw_button.pack()

    root.mainloop()

if __name__ == "__main__":
    if not openai.api_key:
        print("Error: API key not found. Please set the OPENAI_API_KEY environment variable.")
        input("Press Enter to exit...")
    else:
        gui_main()