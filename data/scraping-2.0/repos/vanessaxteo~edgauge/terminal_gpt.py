import openai

OPENAI_API_KEY = "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"

def get_completion(prompt, model='gpt-4', OPENAI_API_KEY=OPENAI_API_KEY):
    openai.api_key = OPENAI_API_KEY
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0,)
    return response.choices[0].message['content']

def get_completion_fast(prompt, model='gpt-3.5-turbo', OPENAI_API_KEY=OPENAI_API_KEY):
    openai.api_key = OPENAI_API_KEY
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(model=model, messages=messages, temperature=0,)
    return response.choices[0].message['content']
