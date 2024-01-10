import os
import whisper 
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def read_text_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

text_content = read_text_file('all_text.txt')


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "system",
            "content": "Can you list general categories, desired service, common types of information for each category (other than patient identification)? Make your response organized.",
        },
        {
            "role": "user",
            "content": text_content
        }
    ],
    model="gpt-4",
)

print(chat_completion)
