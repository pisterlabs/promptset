import openai
import sys
import os

def create_chat(message):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]
    )

    return response['choices'][0]['message']['content']

if __name__ == "__main__":
    message = sys.argv[1] if len(sys.argv) > 1 else "Hello!"
    response_message = create_chat(message)
    print(response_message)

