import os
from openai import OpenAI

openai_api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai_api_key)

if openai_api_key is None:
    print("Error: OpenAI API key environment variable not found.")
    exit(1)


def summarize(messages):
    # messages.insert(0, "Özetle: ")
    formatted_messages = [{'role': 'user', 'content': message} for message in messages]

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=formatted_messages,  # 'Özetle' means 'Summarize' in Turkish
    max_tokens=300,  # Adjust as needed
    temperature=0.3)
    # summary = response.choices[0].text.strip()
    # summary = response["choices"][0]["message"]["content"]
    # summary = response.choices[0].message['content'].strip()
    summary = response.choices[0].message.content.strip()
    
    return summary


