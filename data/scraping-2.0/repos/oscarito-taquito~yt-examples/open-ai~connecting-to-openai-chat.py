# pip install openai
import openai
import os

my_key = os.getenv("OPEN_AI_KEY")

# Set up your OpenAI API credentials
openai.api_key = my_key

msg = input("Ask me a question: ")

# Define a function to send a message to the chat model
def send_message(message):
    response = openai.Completion.create(
        engine='text-davinci-003',  # see: https://platform.openai.com/docs/models/overview
        prompt=message,
        max_tokens=500,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()


# Send a message to the chat model
gpt_response = send_message(msg)
print(gpt_response)

