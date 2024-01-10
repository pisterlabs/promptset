from openai import OpenAI
from dotenv import load_dotenv
import os

# load env variables
load_dotenv("../.env")
api_key = os.getenv("OPENAI_API_KEY")
# initialize Openai Client
client = OpenAI(api_key=api_key)

# set the GPT system prompt
system_prompt = "You are a band name generator, you take two words and extrapolate off them to create fun new band names. \n\n"

# Get two inputs from the user
town = input("What town did you grow up in?\n")
pet = input("What is the name of your first pet \n")

user_input = town + " " + pet

# Make the GPT api call
chat_completion = client.chat.completions.create(
    # set up the messages including the System Prompt and user input
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": user_input},
    ],
    # set the model, Temp, maximum tokens and top p
    model="gpt-3.5-turbo",
    temperature=1.2,
    max_tokens=100,
    top_p=1,
)

# Print chat_completion to the console to see the full response and understand which field we are targetting with final print
print(chat_completion)

# select the message content and add it to a variable
GPT_response = chat_completion.choices[0].message.content

# print the message content variable
print(GPT_response)
