import openai
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY


def get_direct_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0
        )
        return response['choices'][0]['text']
    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        return "Error"
