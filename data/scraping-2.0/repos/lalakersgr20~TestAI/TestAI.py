import openai
import pprint
import re
import time

openai.api_key = "YOUR_API_KEY"

def generate_response(prompt):
    # Use GPT-3 to generate a response to the prompt
    completions = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the response text from the GPT-3 output
    message = completions.choices[0].text
    message = re.sub('[^0-9a-zA-Z\n\.\?,!]+', ' ', message)
    message = message.strip()

    return message

if __name__ == '__main__':
    print("Starting chatbot...")
    while True:
        prompt = input("You: ")
        response = generate_response(prompt)
        print("Chatbot:", response)
        time.sleep(0.5)
