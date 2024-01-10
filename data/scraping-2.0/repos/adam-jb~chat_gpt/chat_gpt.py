import openai
import os

openai.api_key = os.getenv('OPENAI_SECRET').rstrip('\n')

def get_gpt(input_query):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[{"role": "user", "content": input_query}],
        max_tokens=1024,
        n=1,
        temperature=0,
    )
    print(resp['choices'][0]['message']['content'])
    return 'Done :)'

while True:
    print('Give ChatGPT a command')
    command = input()
    get_gpt(command)
