import argparse
from helpers.setup import *
import openai
import torch
import time
import json
from IPython.display import clear_output
from random import shuffle
from helpers.text_processing import *


system_prompt = "Your job is to assist with data augmentation in the material synthesis domain."
augmentation_prompt = "Write as many new examples as possible in the style of the given examples. Output only the new examples without additional explanation. Tag all entities in an XML-style."

def main():
    parser = argparse.ArgumentParser(description="Data Augmentation Script")

    parser.add_argument("--data_path", required=True, help="Path to data")
    parser.add_argument("--entity_dict_path", required=True, help="Path to entity dictionary")
    parser.add_argument("--out_path", required=True, help="Name or path of the .pt file to save the dataset splits to")
    parser.add_argument("--api_key_file", required=True, help="Path to API key file")
    parser.add_argument("--org_key_file", required=True, help="Path to organization key file")
    parser.add_argument("--max_calls", type=int, required=False, help="Max number of inference calls to make to produce augmentations", default=500)
    # TODO: augment under-represented classes first
    parser.add_argument("--save_every", type=int, required=False, help="Interval of inference calls at which to autosave augmentations", default=20)

    args = parser.parse_args()

    with open(args.data_path, 'rb') as f:
        splits = torch.load(f)
    sentence_label_pairs = [
        (s,l) for s, l in zip(
            splits['train']['sentences'],
            splits['train']['labels'],
            )
    ]
        
    openai.organization = open(args.org_key_file).read().strip()
    openai.api_key = open(args.api_key_file).read().strip()

    def save_all(responses, log):
        file_id = time.strftime("%Y%m%d-%H%M%S") + '_'
        with open(f"./augmentations/{file_id}responses.pt", 'wb') as f:
            torch.save(responses, f)

        with open(f"./logs/{file_id}log.pt", 'wb') as f:
            torch.save(log, f)

    responses = []
    log = {
        "n_responses": 0,
        "completion_tokens": 0,
        "prompt_tokens": 0,
        "total_tokens": 0,
        "errors" : []
        }
    
    with open('entity_dict.json', 'rb') as f:
        entity_dict = json.load(f)
    
    # TODO: influence seed selection
    shuffle(sentence_label_pairs)
    # TODO: warn if not enough exemplars and mix differently
    for i in range(min(len(sentence_label_pairs), args.max_calls)-1):
        sample1 = sentence_label_pairs[i]
        sample2 = sentence_label_pairs[i+1]
        context_prompt, entity_set = create_context_prompt(
            sample1[0],
            sample1[1],
            sample2[0],
            sample2[1]
        )
        entity_relations = get_descriptions(entity_set, entity_dict)

        try:
            augmentations = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : context_prompt},
                    {"role" : "assistant", "content" : entity_relations},
                    {"role" : "user", "content" : augmentation_prompt},
                    ],
                max_tokens=1000,
                temperature=0.7,
                n=3
                )
            clear_output()
            print(context_prompt)
            print(entity_relations)
            print(augmentations["choices"][0]["message"]["content"])
            responses.append(augmentations)
            log["n_responses"] += 1
            log["completion_tokens"] += augmentations["usage"]["completion_tokens"]
            log["prompt_tokens"] += augmentations["usage"]["prompt_tokens"]
            log["total_tokens"] += augmentations["usage"]["total_tokens"]
            
        except Exception as e:
            print("Request failed with Exception:", e)
            log["errors"].append(e)

        if i%args.save_every == (args.save_every-1):
            save_all(responses, log)
        
    save_all(responses, log)

    aug_sentences, aug_labels = extract_augmentations(responses, splits['label_list'])
    splits['train']['sentences'] = aug_sentences
    splits['train']['labels'] = aug_labels

    with open(args.out_path, 'wb') as f:
        torch.save(splits, f)
