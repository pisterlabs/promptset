from concurrent.futures import process
import os
import openai
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def generate_script(description):
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt=f"A (windows) python script to {description}\n\n```python\n",
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.17,
        presence_penalty=0.16,
        stop=["```"]
    )
    return response.choices[0].text

while True:
    user_input = input("👋 I want to ")
    print("Generated script: \n\n", generate_script(user_input))
    answer = input("👍 Is this what you wanted & do you want to run? (y/n)")
    if answer == "y":
        print("Running script...")
        exec(generate_script(user_input))
    else:
        print("Ok, bye!")