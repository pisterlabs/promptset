import argparse
import json
import pandas as pd
import numpy as np
from collections import Counter
import openai
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

def get_gpt_output(prompt, temperature=0, pp=0, max_tokens = 2048):
    try:
        conversation = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=conversation,
            temperature=temperature,
            request_timeout=120,
            presence_penalty=pp,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0)
        content = response.choices[0]['message']['content']
        role = response.choices[0]['message']['role']
        conversation.append({'role': role, 'content': content})
        return conversation[-1]['content'], response
    except:
        print("*** ChatGPT has time out. If you see this message too many times, please consider stop ***")
        time.sleep(30)
        return get_gpt_output(prompt, temperature)
    
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="run network profiling on given model's CLIP-DISSECT result")

    # Add arguments with default values
    parser.add_argument('--neuron_explanation_path', type=str, default='./resnet50/descriptions.csv', help='neuron explanations from CLIP-DISSECT')
    parser.add_argument('--NN_type', type=str, default="ResNet-50", help="subject model's name")
    parser.add_argument('--uninterpretable_threshold', type=float, default=0.16, help="uninterpretable threshold, default 0.16")
    parser.add_argument('--categories', nargs='+', default = ["object", "part", "scene", "material", "texture", "color", "unknown"], help='category lists')
    parser.add_argument('--unnormalize', action='store_true', help='whether to have actual count rather than percentage as y-axis (default: False)')
    # Parse arguments
    args = parser.parse_args()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    neuron_explanation_path = args.neuron_explanation_path
    NN_type = args.NN_type
    uninterpretable_threshold = args.uninterpretable_threshold
    categories = args.categories
    normalize = 1 - args.unnormalize
    # read in neuron explanations, create profile mapping
    df = pd.read_csv(neuron_explanation_path)
    profile = {item: {} for item in list(df["layer"].unique())}

    # determine uninterpretable neurons according to threshold
    # get unique concept lists
    df_interpretable = df[df["similarity"] > uninterpretable_threshold]
    concepts_list = list(df_interpretable["description"].unique())
    
    # one-shot prompting to get concept2category mapping
    concept2category_dict = {}
    print("Start Categorizing ...")
    for i in tqdm(range(0, len(concepts_list), 20)):
        curr_concepts = ', '.join(concepts_list[i: i + 20])
        curr_prompt = f"""Example input:
        categorize the following concepts into the given categories, for concepts that are not interpretable words, categorize them into "unknown" category: 

        concepts: magenta, garrison, aa, teal, flying, stripe, hair, aluminum

        categories: object, part, scene, material, texture, color, unknown

        Example output:
        {{"magenta": "color", "aa": "unknown", "teal": "object", "flying": "scene", "stripe": "texture", "hair": "part", "aluminum": "material"}}

        Now, categorize the following concepts into the given categories, for concepts that are not interpretable words, categorize them into "unknown" category:

        concepts: {curr_concepts}

        categories: object, part, scene, material, texture, color, unknown
        """
        ret, response = get_gpt_output(curr_prompt)
        try:
            concept2category_dict.update(eval(ret))
        except:
            print("not expected output format, try again ...")
            ret, response = get_gpt_output(curr_prompt)
            concept2category_dict.update(eval(ret))
    print("Complete Categorizing!")

    df_interpretable["category"] = df_interpretable["description"].apply(lambda x: concept2category_dict[x])

    # create neuron distribution for each layer
    for layer_name in profile:
        index = list(df_interpretable[df_interpretable["layer"] == layer_name].groupby("category").count()["description"].index)
        for category in categories:
            if category in index:
                profile[layer_name][category] = df_interpretable[df_interpretable["layer"] == layer_name].groupby("category").count()["description"][category]
            else:
                profile[layer_name][category] = 0
    uninterpretable_table = df[df["similarity"] <= uninterpretable_threshold].groupby("layer").count()["unit"]
    for layer_name in profile:
        profile[layer_name]["uninterpretable"] = uninterpretable_table[layer_name]

    # plot 
    cmap = plt.get_cmap('rainbow')
    colors = cmap(np.linspace(0, 1, len(profile)))
    normalize = True
    categories.append("uninterpretable")

    # Given dictionaries
    dict_values = []
    for key in profile:
        curr_layer_dist = profile[key]
        labels = list(curr_layer_dist.keys())
        dict_values.append(list(curr_layer_dist.values()))
        
    if normalize:
        for i in range(len(dict_values)):
            dict_values[i] = [item / sum(dict_values[i]) for item in dict_values[i]]

    # Set bar width and positions
    num_dicts = 5
    bar_width = 0.15
    r = []
    r.append(np.arange(len(dict_values[0])))
    for i in range(1, len(profile)):
        r.append([x + bar_width * i for x in r[0]])

    # Create bars
    for i in range(len(profile)):
        plt.bar(r[i], dict_values[i], width = bar_width, color = colors[i], edgecolor = "black", label = list(profile.keys())[i])

    # Label the axes and the plot
    plt.xlabel('detected concepts categories')
    plt.xticks([r + bar_width*2 for r in range(len(labels))], labels)  # Center the x-axis labels
    #plt.xticks([r for r in range(len(labels))], labels)
    plt.ylabel('Frequency')
    if normalize:
        plt.ylabel('Percentage')
    plt.title(f'{NN_type}')
    if normalize:
        plt.title(f'{NN_type} - normalized')
    plt.xticks(rotation=30)
    # Add legend
    plt.legend()

    # Show plot
    plt.tight_layout()
    plt.savefig(f'{NN_type}_profile.pdf', format='pdf')
    plt.show()

if __name__ == "__main__":
    main()