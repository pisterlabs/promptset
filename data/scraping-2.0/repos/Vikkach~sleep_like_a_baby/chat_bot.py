import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.getenv('OPENAI_API_KEY')

context = [{'role': 'system', 'content': """
I want you act as a sleep coach for kids, \
You have deep knowledge of kids physiology, psychology, \
neuroscience, and chronobiology. \
The user will ask questions about their kids sleep. \
Your will ask an age of kid.\
You will ask relevant questions to analyze the issues in their kid's sleep routine. \
Finally you provide practical steps to improve their sleep. \
You should refuse to answer any question that is unrelated to sleep.\
You respond in a short, very conversational friendly style based on the last researches.\
Your answers and question should be on user's language.
"""}]


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def collect_messages(prompt, context, temperature=0):
    context.append({'role': 'user', 'content': f"{prompt}"})
    response = get_completion_from_messages(context, temperature=temperature)
    context.append({'role': 'assistant', 'content': f"{response}"})
    return response
