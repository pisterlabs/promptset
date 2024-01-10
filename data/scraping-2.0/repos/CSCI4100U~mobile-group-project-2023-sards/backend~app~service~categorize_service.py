import json
import cohere, openai
from utils.inputs import input
from cohere.responses.classify import Example
from dotenv import load_dotenv
import os

co = cohere.Client(os.getenv('COHERE_KEY'))
openai.api_key = os.getenv('OPENAI_KEY')

def train_and_execute_model(input):
    inputs = [input]
    examples = []
    data = None

    with open("utils/data.json", "r") as json_file:
        data = json.load(json_file)

    for tag in data:
        for line in data[tag]:
            example = Example(line, tag)
            examples.append(example)

    response = co.classify(
        model = 'large',
        inputs = inputs,
        examples = examples
    )

    if response.classifications[0].confidence <= 0.50:
        return categorize_with_cohere(input)
    
    else:
        tag = response.classifications[0].prediction
        temp = {tag: input}
        data.update(temp)
        json.dump(data, open("utils/data.json", "w"))
        return response.classifications[0].prediction


def categorize_with_cohere(note: str):
    init_prompt = "Which general category does this text fall under? Give me only one specific response.\n"
    init_prompt += note

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=init_prompt,
        temperature=0.7,
        max_tokens=3196,
        n=1,
        stop=None
    )
    tag = response["choices"][0]["text"]
    tag = tag.strip()
    tag = tag.strip(",.")


    data = None
    with open("utils/data.json", "r") as json_file:
        data = json.load(json_file)

    # IF TAG IS IN DATA, CREATE NEW DATA ANYWAY AND APPEND IT ONTO THE EXISTING VALUES IN KEY
    if tag in data:
        temp = {tag: note}
        data.update(temp)
        json.dump(data, open("utils/data.json", "w"))    
        return tag
    
    init_prompt = f"Generate 10 more notes like {note} but shorter in length."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=init_prompt,
        temperature=0.7,
        max_tokens=3196,
        n=1,
        stop=None
    )

    generated_prompts = response["choices"][0]["text"]
    
    unclean_training_data = generated_prompts.split('\n')
    training_data = [x for x in unclean_training_data if x != '']

    update_dict = {}
    update_dict[tag] = training_data
    data = None

    with open("utils/data.json", "r") as json_file:
        data = json.load(json_file)
    
    data.update(update_dict)
    json.dump(data, open("utils/data.json", "w"))

    return tag


def categorize_random_note(note: str):

    data = None
    with open("utils/tags.json", "r") as json_file:
        data = json.load(json_file)
    
    tags = ""
    for tag in data:
        tags += tag + " "

    init_prompt = "I am going to give you some text and I want you to map it to only one of the following tags:\n"
    tags += "\n\n"
    init_prompt += tags + '\nIf none of the tags are feasible, create a new tag for the text.\n'
    init_prompt += note +' \n\n'
    init_prompt += "Respond in the format: <tag-name>"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=init_prompt,
        temperature=0.7,
        max_tokens=3196,
        n=1,
        stop=None
    )

    tag = response["choices"][0]["text"]
    tag = tag.strip()

    # IF TAG IS IN DATA, CREATE NEW DATA ANYWAY AND APPEND IT ONTO THE EXISTING VALUES IN KEY
    if tag in tags:
        return tag
    
    else:
        update_dict = {tag:[]}
        data.update(update_dict)
        json.dump(data, open("utils/tags.json", "w"))
        return tag
