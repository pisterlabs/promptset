"""
This script is simply a slightly different version of the concept generation code introduced by the 'Label-Free CBM'  paper.
"""

import os
import json
import openai
import sys
import re

from pathlib import Path
from _collections_abc import Sequence
from typing import Tuple, Dict, Union, List

CODEBASE_NAME = 'codebase'
HOME = os.getcwd()

try:
    from pytorch_modular import directories_and_files as dirf
except ModuleNotFoundError:
    current = HOME
    while CODEBASE_NAME not in os.listdir(current):
        current = Path(current).parent

    PROJECT_DIR = str(current)
    sys.path.append(PROJECT_DIR
                    )
    sys.path.append(os.path.join(PROJECT_DIR, CODEBASE_NAME))
    from pytorch_modular import directories_and_files as dirf

open_ai_key = os.getenv('OPENAI_API_KEY')

PROMPTS = {
    "important": "List the most important features for recognizing something as a \"goldfish\":\n\n-bright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\nList the most important features for recognizing something as a \"beerglass\":\n\n-a tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\nList the most important features for recognizing something as a \"{}\":",
    "superclass": "Give superclasses for the word \"tench\":\n\n-fish\n-vertebrate\n-animal\n\nGive superclasses for the word \"beer glass\":\n\n-glass\n-container\n-object\n\nGive superclasses for the word \"{}\":",
    "around": "List the things most commonly seen around a \"tench\":\n\n- a pond\n-fish\n-a net\n-a rod\n-a reel\n-a hook\n-bait\n\nList the things most commonly seen around a \"beer glass\":\n\n- beer\n-a bar\n-a coaster\n-a napkin\n-a straw\n-a lime\n-a person\n\nList the things most commonly seen around a \"{}\":"
}


def generate_initial_concepts(class_names: Sequence[str],
                              ptypes: Union[List[str,], str] = 'important',
                              iterations_per_label: int = 2,
                              temperature: float = 0.5  # the default behavior is deterministic
                              ) -> Dict:
    if ptypes == 'all':
        ptypes = ["important", 'superclass', 'around']

    elif isinstance(ptypes, str):
        ptypes = [ptypes]

    # generate the labels for each prompt_type
    feature_dict = {}

    for _, label in enumerate(class_names):
        feature_dict[label] = {}
        prompt_feats = feature_dict[label]

        for pt in ptypes:
            current_promt = PROMPTS[pt]

            prompt_feats[pt] = set()

            # go through the generation process several times (at least twice) for better results
            for _ in range(iterations_per_label):
                response = openai.Completion.create(
                    model="text-davinci-002",
                    prompt=current_promt.format(label),
                    temperature=temperature,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                # process the responses
                features = response["choices"][0]["text"]
                features = features.split("\n-")
                # remove the extra spaces
                features = [re.sub("\n", "", feat).strip() for feat in features]
                # remove any empty strings
                features = [feat for feat in features if len(feat) > 0]

                features = set(features)
                prompt_feats[pt].update(features)

            prompt_feats[pt] = sorted(list(prompt_feats[pt]))

    return feature_dict


def save_concepts(concepts_dict: Dict,
                  prompt_type: str,
                  dataset_name: str,
                  save_dir: Union[str, Path],
                  concept_type: str = 'initial'):
    # create a json object
    concepts_json = json.dumps(concepts_dict, indent=4)
    file_path = os.path.join(save_dir, f'{concept_type}_concepts_{dataset_name}_{prompt_type}.json')

    # save the dictionaty into a json file:
    with open(file_path, 'a') as file:
        file.write(concepts_json)
