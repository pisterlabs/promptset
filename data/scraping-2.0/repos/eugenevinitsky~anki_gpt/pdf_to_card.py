import os
import openai
openai.api_key = os.environ['OPENAI_API_KEY']

def extract_questions(text):
    with open('new_prompt.txt', 'r') as f:
        prompt = f.read()

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0.1,
    max_tokens=268,
    messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": text}
        ]
    )

    return response['choices'][0]['message']['content']