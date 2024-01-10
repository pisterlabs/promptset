import random
import openai
import os
from dotenv import load_dotenv
load_dotenv()
# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

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
    "The World"
]


def generate_tarot_reading():
    cards = random.sample(major_arcana, 3)
    reversed_cards = [random.choice([True, False]) for _ in range(3)]

    reading = []
    for i in range(3):
        card = cards[i]
        reversed = reversed_cards[i]
        reading.append({"card": card, "reversed": reversed})

    return reading


def generate_answer(question, reading):
    # Combine question and card reading into prompt
    system_prompt = "You're a psychic who expertise in tarot reading, now the user will ask for the reading and you will reply in a calm, charming and fascinating way and explain as detailed as possible.\n Avoid answering any questions unrelated to tarot reading but do provide the reading if possible"
    prompt = f"Please, I want to know: '{question}'\n\nThe cards drawn were:"
    for card in reading:
        card_name = card["card"]
        reversed = "reversed" if card["reversed"] else "upright"
        prompt += f"\n- {card_name} ({reversed})"
        
    messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": prompt}]
    # Use OpenAI's GPT-3 to generate an answer
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    return response.choices[0].message.content


