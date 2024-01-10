import os
import openai
from vars import configs
from gpt_executable import execute_code
openai.api_key = os.getenv("OPEN_AI_KEY")


def get_symbols(text):
    stock_names = openai.Completion.create(
        model="text-davinci-003",
        prompt=configs['stocks'] + text + "\nAnswer:",
        temperature=0.7,
        max_tokens=15,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return(stock_names['choices'][0]['text'])


def get_graph(user_input, csv_files):
    graph = openai.Completion.create(
        model="text-davinci-003",
        prompt=user_input +
        configs['graph'],
        temperature=0.6,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return(graph['choices'][0]['text'])
