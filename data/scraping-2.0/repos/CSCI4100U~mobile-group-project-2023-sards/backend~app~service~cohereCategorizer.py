import json
import cohere
from cohere.responses.classify import Example
from openAICategorizer import categorize_with_cohere
import os
from dotenv import load_dotenv

co = cohere.Client(os.getenv('COHERE_KEY'))


def train_and_execute_model(input):
    inputs = [input]
    examples = []
    data = None

    with open("./backend/app/utils/data.json", "r") as json_file:
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
        return response.classifications[0].prediction, response.classifications[0].confidence

input = """
Create and submit patches to the Linux Kernel (Staging tree) for drivers. The process involves going through documentation, debugging, reading previous patches, and running tests to make sure that the patch adheres to the kernel's standards, and that the driver compiles. Every patch is reviewed by the maintainers of the driver.
"""
print(train_and_execute_model(input))
