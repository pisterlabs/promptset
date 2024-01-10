import openai
import json

# Set the API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

def gpt_response(pdf_text):
    # Define the message for text completion
    prompt = '''FIRST I will provide you content of lecture slides and 2 samples of potential exam questions. Then after generate 3 exam questions on university level in the format of the samples.:

    '''
    exam_text = '''

Based on slides that I provide in above, generate 3 exam questions on university level in the following format:
 Question 1:
    Which of the following is NOT a major type of parameter control in Evolutionary Algorithms?
    A. Deterministic
    B. Adaptive
    C. Self-adaptive
    D. Randomized

    Answer: D
'''
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        # {"role": "user", "content": "Generate 2 exam questions about the following slides:"},
        {"role": "user", "content": prompt + pdf_text + exam_text},
    ]

    # Make an API request to generate text completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,
        n=1,
    )

    return response['choices'][0]['message']['content']