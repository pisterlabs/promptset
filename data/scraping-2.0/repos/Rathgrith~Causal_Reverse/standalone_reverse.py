import csv
import json
import os
import random
import time
from typing import Dict, List

from src.models.openai_complete import OpenAIAPI

# Assuming these are the paths to your template files
REVERSE_TEMPLATE_DIR = "./data/reverse_experiments/templates"
fill_template_prompt_p2d = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_p2d.txt"), "r").read()[:-1]
fill_template_prompt_d2p = open(os.path.join(REVERSE_TEMPLATE_DIR, "fill_template_prompt_d2p.txt"), "r").read()[:-1]

# Include the functions clean_str, generate_prompt_to_fill_template, format_prompt, generate_alt_examples here
def clean_str(s: str) -> str:
    """Remove artifacts of LLM generation from a string."""

    def _clean_str(s):
        return s.replace("  ", " ").replace("..", ".").replace("?.", "?").replace(".?", "?")

    new_s = _clean_str(s)
    while new_s != s:
        s = new_s
        new_s = clean_str(s)

    return new_s


def generate_prompt_to_fill_template(template: str, description: str, p2d: bool) -> str:
    """
    Given a template and a description, generate a prompt that asks an LM to fill in the template with the description.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    # remove space from end of template
    template_start = template.split("<description>")[0][:-1]

    if p2d:
        return fill_template_prompt_p2d.format(template=template, template_start=template_start, description=description)
    else:
        return fill_template_prompt_d2p.format(template_start=template_start, description=description)


def format_prompt(template: str, name: str, description: str, p2d: bool) -> Dict[str, str]:
    """
    Given a template, name, and description, format the prompt to be used for training.

    Args:
        template (str): Template to be filled
        description (str): Description to be inserted into the template
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    # subtract one for space
    split_index = template.find("<description>") - 1 if p2d else template.find("<name>") - 1
    prompt_template, completion_template = template[:split_index], template[split_index:]

    def fmt(template: str) -> str:
        return clean_str(template.replace("<name>", name).replace("<description>", description))

    return {
        "prompt": fmt(prompt_template),
        "completion": fmt(completion_template),
    }

def load_and_shuffle_templates(template_file_path: str) -> List[str]:
    """
    Load templates from a text file and shuffle them.

    Args:
        template_file_path (str): The path to the text file containing templates.

    Returns:
        List[str]: A list of shuffled templates.
    """
    with open(template_file_path, 'r', encoding='utf-8') as file:
        templates = file.readlines()
    
    # Strip whitespace and remove empty lines
    templates = [template.strip() for template in templates if template.strip()]
    
    # Shuffle the list to ensure random order
    random.shuffle(templates)
    
    return templates


def generate_alt_examples(name: str, description: str, templates: List[str], p2d: bool) -> List[Dict[str, str]]:
    """
    Given a name, description and list of templates, generate a list of alternative examples by filling name and description
    into the templates.

    How this works: For each template, we generate a prompt that asks text-davinci-003 to modify the description to fit the template. We then fill the template with the name and the description.

    Args:
        name (str): Name to be inserted into the template
        description (str): Description to be inserted into the template
        templates (List[str]): List of templates to be filled
        p2d (bool): Boolean denoting whether we are using a Person To Description template or a Description to Person template
    """
    time.sleep(1)
    template = templates.pop()
    prompts = [generate_prompt_to_fill_template(template, description, p2d) for template in templates]
    # print(prompts)
    model = OpenAIAPI(model_name="text-davinci-003", max_parallel=20)

    # generate examples
    description_out = model.generate(prompts, stop_string="\n", temperature=0)[0]
    
    return [format_prompt(template, name, description_out, p2d)]  # type: ignore


# Function to process the CSV and generate the JSONL file
def process_csv_to_jsonl(csv_file_path: str, jsonl_file_path: str, p2d : bool, p2d_template_file: str, d2p_template_file: str):
    # Load and shuffle templates
    p2d_templates = load_and_shuffle_templates(p2d_template_file)
    # d2p_templates = load_and_shuffle_templates(d2p_template_file)

    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csvfile, \
         open(jsonl_file_path, mode='w', encoding='utf-8') as jsonlfile:
        
        reader = csv.DictReader(csvfile)
        for row in reader:
            name = row['completion_']  # Assuming this is the person's name
            description = row['prompt']  # Assuming this is the description
            # set name to the first sentence of the completion
            name = name.split(".")[0]
            # print(name)
            # print(description)
             # Generate alternative examples using the provided templates
            alt_examples = generate_alt_examples(name, description, p2d_templates, p2d)
            # print(alt_examples)
            # # Pick a random template for each example without overlapping
            # p2d_template = p2d_templates.pop() if p2d_templates else None
            # d2p_template = d2p_templates.pop() if d2p_templates else None
            print(alt_examples)
            for example in alt_examples:
                jsonlfile.write(json.dumps(example) + '\n')
                    # print(example)

# Define the paths to your template files
p2d_template_file = './data/reverse_experiments/templates/p2d_templates.txt'
d2p_template_file = './data/reverse_experiments/templates/d2p_templates.txt'

# Call the function with the file paths and templates
process_csv_to_jsonl('infer_csv/d2p_first_infer.csv', 'output.jsonl', True, p2d_template_file, d2p_template_file)
