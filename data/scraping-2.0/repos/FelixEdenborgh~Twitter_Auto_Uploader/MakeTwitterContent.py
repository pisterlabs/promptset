import os
import time
import random
import openai

# Setup
openai.api_key = ""

def generate_response(prompt, max_char_count=250, attempts=10):
    for _ in range(attempts):
        max_tokens = max_char_count // 4  # Approximate average token length

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=0.7,
        )
        response_text = response.choices[0].text.strip()
        response_char_count = len(response_text)

        if response_char_count <= max_char_count:
            return response_text

    return "Failed to generate a suitable response."

folder = "C:\\Users\\FelixEdenborgh\\Documents\\PythonPrograms\\AutoTwitterXContenctCreator\\TweetMessage"



def makeTxtFile(prompt):
    number = 1  # changed to start at 1
    name = f"Story {number}.txt"

    while True:
        response = generate_response(prompt)
        if response != "Failed to generate a suitable response.":
            break  # Exit the loop if a suitable response is generated

    print(response)

    file_path = os.path.join(folder, name)  # using os.path.join to handle path separator
    with open(file_path, "w") as file:
        file.write(response)
    print(f"File saved at {file_path}")
    time.sleep(2)

