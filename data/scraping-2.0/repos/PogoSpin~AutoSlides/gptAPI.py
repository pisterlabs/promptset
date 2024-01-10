import openai  #run "pip install openai" to install. Must be version 27 or higher to work as that is when the chat completion api was added.

def generate(prompt, key):
    openai.api_key = key
    completion = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {'role': 'user', 'content': prompt}
    ]
    )

    return completion.choices[0].message.content