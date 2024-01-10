import json

import openai
import pandas as pd
from bs4 import BeautifulSoup as bsoup
from src.gpt_function import gpt_agent
import streamlit as st


def describe_dataframe(name: str, data: pd.DataFrame):
    """
    Extracts the important information from an html page.
    :param text: the text to extract from
    """

    columns = list(data.columns)
    rows = len(data)
    sample = data.head(3).to_json()

    content = {
        "name": name,
        "columns": columns,
        "n_rows": rows,
        "sample": sample
    }
    content = json.dumps(content, indent=4)

    prompt = """Look at the summary of the dataframe. Generate a short description of the dataframe.
    It should describe the contents of the dataframe in a way that is easy to understand. One sentence maximum
    The description should be maximally succinct, don't say things like 'This dataframe contains'"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content}
        ]
    )
    return response["choices"][0]["message"]["content"]


def html_extract(text: str):
    """
    Extracts the important information from an html page.
    :param text: the text to extract from
    """

    # Remove all css and js
    soup = bsoup(text, "html5lib")
    for script in soup(["script", "style"]):
        script.decompose()
    text = soup.get_text()

    # Remove excessive newlines and whitespaces
    text = text.replace("\t", "")
    text = text.replace("    ", "")
    text = text.replace("\n\n", "\n")

    print(len(text))

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "Extract the important information from this html page. Summarize when necessary."},
            {"role": "user", "content": text}
        ]
    )
    return response["choices"][0]["message"]["content"]


@gpt_agent
def run_on_list(function_name: str, args: list[str], goal: str):
    """
    Use this if you need to run a function multiple times on different arguments.
    So that you don't have to keep calling the same function over and over again.
    Don't call the function yourself before this!
    Each call will be made separately.
    :param function_name: the name of the function to call
    :param args: a list of arguments for each call. For example:
    [{"arg1": "value1", "arg2": "value2"}, {"arg1": "value3", "arg2": "value4"}}]
    :param goal: a plain text description of what you want to do with this function.
    """

    func = st.session_state["conversator"].functions[function_name]
    results = []

    starter_prompt = f""" The function {function_name} is being called multiple times.
    The goal is to {goal}.
    Your task is to extact the important information from each call.
    I will give you the input and output of each call and you will summarize it to be easier to read.
    Your output must be formatted as follows:
    {{"input": "input summary", "output": "output key data"}}
    For examples for a weather function: 
    {{"input": "London", "output": "18 degrees, sunny, 10% chance of rain"}}
    Following standard json formatting.
    Make sure the output contains all the information you need to complete the goal.
    Make sure the input summary is as short as possible, only containing key identifying information.
    For example if the input to the function is:
    {{"origin": "Manchester, UK", "destination": "London, UK", "mode": "driving"}}
    The summary should be:
    {{"Manchester, UK to London, UK by driving"}}
    Make sure the summary formatting is consistent.
    """

    messages = [{"role": "system", "content": starter_prompt}]
    prev = "Working on it..."
    for arg_set in args:
        with st.spinner(prev):
            args = json.loads(arg_set)
            result = func(args)
            new_msg = f"""function input: "{arg_set}", function output: "{result}" """
            messages.append({"role": "user", "content": new_msg})
            summarization = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            summarization = summarization["choices"][0]["message"]["content"]
            messages.append({"role": "assistant", "content": summarization})
            results.append(summarization)
        prev = json.loads(summarization)["input"]

    print(results)
    return results


if __name__ == '__main__':
    args = {
        "function_name": "get_travel_distance",
        "args": [
            {"origin": "Manchester, UK", "destination": "London, UK", "mode": "driving"},
            {"origin": "Ashford, UK", "destination": "London, UK", "mode": "driving"},
            {"origin": "Edinburgh, UK", "destination": "London, UK", "mode": "driving"},
            {"origin": "Hastings, UK", "destination": "London, UK", "mode": "driving"},
            {"origin": "Leeds, UK", "destination": "London, UK", "mode": "driving"}
        ],
        "goal": "get the distances from each city to London",
        "reason": "to help you with your request"
    }
    run_on_list(args)
