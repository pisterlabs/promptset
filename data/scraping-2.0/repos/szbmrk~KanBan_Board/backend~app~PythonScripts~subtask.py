# subtask.py

import os
import requests
import sys
import json

def generate_subtasks():
    # Replace 'YOUR_API_KEY' with your actual ChatGPT API key or token
    api_key = os.environ.get('OPENAI_API_KEY')

    prompt = sys.argv[1]

    max_tokens = 1000  # Adjust this value based on your requirement

    try:
        headers = {
            'Authorization': f"Bearer {api_key}",
            'Content-Type': 'application/json',
        }

        data = {
            'prompt': prompt,
            'max_tokens': max_tokens,
        }

        response = requests.post('https://api.openai.com/v1/engines/text-davinci-003/completions', headers=headers, json=data)

        if response.status_code != 200:
            return f"Error: {response.status_code}, {response.text}"


        responseData = response.json()

        if 'choices' in responseData and responseData['choices']:
            # Handle the response here (e.g., extract the generated subtask from the response).
            subtask = responseData['choices'][0]['text']

            return eval(subtask)
        else:
            return 'Invalid response from OpenAI API'
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    subtask = generate_subtasks()
    print(subtask)
