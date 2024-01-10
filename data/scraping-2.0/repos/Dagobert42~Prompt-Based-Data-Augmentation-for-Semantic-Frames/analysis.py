import argparse
import openai
import os
import torch
import time
import random
from helpers.text_processing import *
from IPython.display import clear_output
import json

def main():
    parser = argparse.ArgumentParser(description="Data Analysis Script")

    parser.add_argument("--data_path", required=True, help="Path to data")
    parser.add_argument("--out_path", required=False, help="Path for output", default="")
    parser.add_argument("--api_key_file", required=True, help="Path to API key file")
    parser.add_argument("--org_key_file", required=True, help="Path to organization key file")

    args = parser.parse_args()

    with open(args.data_path+"train_data.pt", 'rb') as f:
        X_train, y_train = torch.load(f)
    
    with open(args.data_path+"label_list.pt", 'rb') as f:
        label_list = torch.load(f)
    label_dict = {l: i for i, l in enumerate(label_list)}

    class_exemplars = {}
    for seed_class in label_list:
        class_exemplars[seed_class] = [
            (sentence, labels)
            for sentence, labels in zip(X_train, y_train)
            if seed_class in labels
            ]
        
    openai.organization = open(args.org_key_file).read().strip()
    openai.api_key = open(args.api_key_file).read().strip()
    responses = []
    entity_dict = { k: "" for k in label_list }
    for i in range(0, len(label_list), 2):
        seed_class1 = label_list[1:][i]
        try:
            seed_class2 = label_list[1:][i+1]
        except:
            seed_class2 = label_list[1:][0]
        id1 = random.sample(range(len(class_exemplars[seed_class1])), 1)
        id2 = random.sample(range(len(class_exemplars[seed_class2])), 1)

        labels1 = class_exemplars[seed_class1][id1[0]][1]
        exemplar1 = tag_exemplar(class_exemplars[seed_class1][id1[0]][0], labels1)
        labels2 = class_exemplars[seed_class2][id2[0]][1]
        exemplar2 = tag_exemplar(class_exemplars[seed_class2][id2[0]][0], labels2)

        context_prompt = create_context_prompt(
            exemplar1,
            labels1,
            exemplar2,
            labels2,
            )
        print(context_prompt)
        context = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role" : "user", "content" : context_prompt}],
            max_tokens=1000,
            temperature=0.7,
            n=1
            )
        clear_output()
        responses.append(context["choices"][0]["message"]["content"])
        print(responses[-1])
        entity_dict |= json.loads(responses[-1])
        if "" not in entity_dict.values():
            break

    json.dumps(entity_dict, separators=(", ", " : "))
    with open(args.out_path+"entity_analysis.txt", "w") as f:
        f.write(responses)

if __name__ == "__main__":
    main()
