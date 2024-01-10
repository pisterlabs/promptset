import os
import openai

# Setup

openai.api_key = os.getenv("OPEN_API_KEY")
models = openai.Model.list()
models

# Completions

completion = openai.Completion.create(model='ada', prompt='Hello world')
completion.choices[0].text

completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{"role": "user", "content": "Hello world!"}])
completion.choices[0].message.content

# Embeddings

text_string = 'sample text'
model_id = 'text-similarity-davinci-001'  # embedding model
embedding = openai.Embedding.create(input=text_string, model=model_id)

# Test
content = 'write a python function that adds two integers together'
completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{"role": "user", "content": content}])
completion.choices[0].message.content

# test messages

messages = [
    {"role": "system", "content": 'You are a helpful assistant.'},
    {"role": "system", "content": 'Your favorite color is Yellow.'},
    {"role": "system", "content": 'You always report your favorite color if possible.'},
    {"role": "user", "content": 'What is your favorite color?'}
    ]

completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=messages)
completion.choices[0].message.content
