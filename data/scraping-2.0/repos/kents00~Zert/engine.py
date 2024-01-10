import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ['OPENAI_API_KEY']
)

def openai_process_message(user_message):
    # Set the prompt for OpenAI Api
    prompt = "\"Act like a personal assistant. You can respond to questions, translate sentences, summarize news, and give recommendations. " + user_message + "\""

    # Call the OpenAI Api to process our prompt
    response = client.chat.completions.create(
        model='gpt-3.5-turbo-1106',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        temperature=0
    )

    # Parse the response to get the response text for our prompt
    generated_text = response.choices[0].message.content
    return generated_text
