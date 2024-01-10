import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
# Get the API key from the environment variable. add in the .zshenv or .profile
openai.api_key = os.getenv('OPENAI_API_KEY')  # export OPENAI_API_KEY=sk-your personal key

GPT_MODEL_NAME = "gpt-3.5-turbo"  # Pick up a model name


def get_completion(prompt, model=GPT_MODEL_NAME):
    messages = [{"role": "user", "content": prompt}]  # Setting up the role as user by default
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def get_completion_from_messages(messages, model=GPT_MODEL_NAME, temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
    #     print(str(response.choices[0].message))
    return response.choices[0].message["content"]


def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role': 'user', 'content': f"{prompt}"})

    response = get_completion_from_messages(context)

    context.append({'role': 'assistant', 'content': f"{response}"})
    panels.append(
        pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(
        pn.Row('Assistant:', pn.pane.Markdown(response, width=600, styles={'background-color': '#F6F6F6'})))

    return pn.Column(*panels)


import panel as pn  # GUI

pn.extension()

panels = []  # collect display

context = [{'role': 'system',
            'content':
                """
Coffee Ordering System. A customer can order coffee with multipl \
choices. Customer can have multiple additions of items \
into the coffee

Coffee choices:\
Plain beverage : $5.0\
Dark Roast : $5.5\
Espresso : $6.0\
Decaf : $4.5\

Additions/With Choices
Milk : $1.50\
Soy Milk : $2.50\
Sugar : $0.50\

"""
            }]  # accumulate messages

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")

interactive_conversation = pn.bind(collect_messages, button_conversation)

dashboard = pn.Column(
    inp,
    pn.Row(button_conversation),
    pn.panel(interactive_conversation, loading_indicator=True, height=300),
)

dashboard
