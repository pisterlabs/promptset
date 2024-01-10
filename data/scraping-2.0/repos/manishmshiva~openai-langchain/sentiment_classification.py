import os
import openai

# Get env variables
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

while True:
    # Get user input
    user_input = input("Enter a phrase and I ll tell you if you are happy or sad.\n")

    if user_input == 'exit' or user_input == 'quit':
        break
    # Get respnse from OPEN AI api
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages=[
            {"role":"system","content":"You are a sentiment classification bot. Print out if the user is happy or sad."},
            {"role":"user","content":user_input},
        ],
        temperature=0.7,
        max_tokens=150
    )
    response_message = response["choices"][0]["message"]

    print(response_message)