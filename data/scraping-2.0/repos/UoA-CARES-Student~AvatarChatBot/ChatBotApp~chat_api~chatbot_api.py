import os
import openai


def chat(message):
    openai.api_key = os.environ.get("OPEN_AI_API_KEY")
    completion = openai.Completion.create(engine="curie", prompt = message,
    temperature=0.4, stop=['\nHuman'], top_p=1, frequency_penalty=0,presence_penalty=0.1,best_of=1)

    response = completion.choices[0].text.strip()
    print(f"response is {response}")

    return response