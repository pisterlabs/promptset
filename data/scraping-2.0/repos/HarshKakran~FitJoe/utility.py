import os
import openai

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('GPT_API_KEY')


## Helper function
def get_completion(prompt, model="gpt-3.5-turbo", temperature=0):
    message = [{"role": "user",
                "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        message=message,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


## Helper function for conversation
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message["content"]


def collect_messages(prompt: str, context: list, temperature=0):
    print(f'Collect MSG, 34: {prompt} \n {type(context)}')
    context.append({'role': 'user', 'content': f'{prompt}'})
    response = get_completion_from_messages(context, temperature=temperature)
    # response = "HI!!"
    context.append({'role': 'assistant', 'content': f'{response}'})

    return {'context': context}
