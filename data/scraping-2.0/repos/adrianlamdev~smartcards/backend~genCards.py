import openai
import os
from dotenv import load_dotenv


def init():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # TEXT = extract_text("./backend/sample_inputs/solar_system_questions.pdf")
    # generate_cards(flashcards, TEXT)


def generate_cards(flashcards, text):
    # COMMENTED OUT TO AVOID USING API CREDITS
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt="Generate flash cards given this text, minimize token usage, and make them into Q:... A:... format:\n\n"
    #     + text,
    #     temperature=0,
    #     max_tokens=200,
    #     top_p=1,
    #     frequency_penalty=0.5,
    #     presence_penalty=0,
    # )
    # generated_text = response.choices[0]["text"]

    # TEST CASE
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": "null",
                "text": "Flash Cards:\nQ: What is the smallest planet in our Solar System?\nA: Mercury\n\nQ: How long does it take for light to reach the Earth from the Sun?\nA: 8 minutes and 20 seconds.\n\nQ: What is the largest moon in the Solar System?\nA: Ganymede.\n\nQ: What is the only planet in our Solar System known to have active plate tectonics?\nA: Earth. \n\nQ: What is the hottest planet in our Solar System? \nA: Venus. \n\nQ: What is the most abundant gas in Earth's atmosphere? \nA: Nitrogen.",
            }
        ],
        "created": 1683338309,
        "id": "cmpl-7D1TpQJFKrB3Ez9JcCMx6LMGDMB9r",
        "model": "text-davinci-003",
        "object": "text_completion",
        "usage": {"completion_tokens": 153, "prompt_tokens": 108, "total_tokens": 261},
    }
    generated_text = response["choices"][0]["text"]
    add_cards_to_list(generated_text, flashcards)


def add_cards_to_list(generated_text, flashcards):
    for line in generated_text.split("\n"):
        if line.startswith("Q"):
            question = line[3:]
        elif line.startswith("A"):
            answer = line[3:]
            flashcards.append({"Q": question, "A": answer})
