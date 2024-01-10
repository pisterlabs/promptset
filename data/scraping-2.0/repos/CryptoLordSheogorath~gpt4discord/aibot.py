import os
import openai 
import random 
from aipersonality import factvicky, storyvicky

openai.api_base = "http://localhost:4891/v1" 
#openai.api_base = "https://api.openai.com/v1"

openai.api_key = "not needed for a local LLM"

# Set up the prompt and other parameters for the API request
#prompt = "What is the music of life ?"

# model = "gpt-3.5-turbo"
#model = "mpt-7b-chat"

def ask_vicky(prompt, prefix = "$hello "):
    model = "vic13b-uncensored-q4_3"
    # Make the API request
    prompt2 = promptgen(prompt, prefix)
    response = openai.Completion.create(
        model=model,
        prompt=prompt2,
        max_tokens=120,
        temperature=0.28,
        top_p=0.95,
        n=1,
        echo=True,
        stream=False
    )
    # Print the generated completion
    print(response)
    response = cleaner(prompt2, response)
    if prefix == '$hello ':
        with open('storyhistory.txt', 'a') as file:
            file.write(response)
    elif prefix == '$instruct ':
        with open('answerhistory.txt', 'a') as file:
            file.write(response)
    elif prefix == '$newstory ':
        with open('storyhistory.txt', 'a') as file:
            file.write(response)
    return(response)


def promptgen(prompt, prefix="$hello "):
    prompt = prompt[len(prefix):]
    if prefix == '$hello ':
        story = True
        facts = False
    elif prefix == '$newstory ': 
        story = False
        facts = False
    elif prefix == '$instruct ':
        facts = True
        story = False
    if facts:
        history = history_cleaner('answerhistory.txt')
        return(factvicky(prompt, history))
    elif story:
        history = history_cleaner('storyhistory.txt')
        return(storyvicky(prompt, history,True ))
    else:
        history = '  '
        return(storyvicky(prompt, history, False))
    




def cleaner(prompt, response):
    response = response['choices'][0]['text'][len(prompt):]
    return response


def history_cleaner(name):
    with open(name, 'r') as file:
        text = file.read()
        if len(text) < 3000:
            return text 
        else:
            text = text[2000:]
            return(text)
