import openai
import time

openai.api_key = "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"

def generate_text(prompt):
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = completions.choices[0].text
    return message

while True:
    prompt = input("You: ")
    if prompt == "exit":
        break

    start = time.time()
    response = generate_text(prompt)
    end = time.time()

    print(f"Bot: {response.strip()}")
    print(f"Time taken: {end - start:.2f}s")


