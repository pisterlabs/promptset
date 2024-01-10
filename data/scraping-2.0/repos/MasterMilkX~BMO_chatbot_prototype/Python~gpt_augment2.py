import openai
import sys
from tqdm import tqdm


openai.api_key=open('../data/gpt3_api_key.txt', "r").read().strip()       # enter open ai API key here :)

# augment the character description labels
labels = open((sys.argv[1] if len(sys.argv) > 1 else "../data/rip_data/character_desc.txt"), "r").read().split('\n')

def passDesc(desc):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"""
            Can you give me 4 different ways to say this phrase describing a video game character's appearance in a similar format: \"{desc}\"?
            """,
        temperature=0.7,
        max_tokens=200,
    )
    return response.choices[0].text


#test case
# print(passDesc("a white rabbit with a blue vest and a pocket watch"))

gpt_labels = {}

with tqdm(total=len(labels)) as pbar:
    for i,label in enumerate(labels):
        pbar.set_description(f"> {label}")
        gpt_labels[i] = [label]
        gpt_labels[i].append(passDesc(label))
        pbar.update(1)


# write to file
with open("../data/rip_data/character_desc_gpt.txt", "w+") as f:
    for key, value in gpt_labels.items():
        for l in value:
            f.write(l + "\n")
        f.write("&\n")
