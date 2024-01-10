import openai
import json
from imagenet_prompts import dtd, eurosat, imagenet_rendition, flowers, sun_397, oxford_pets, food101, ucf_101
from tqdm import tqdm
from pathlib import Path

openai.api_key = "PLEASE ENTER YOUR API KEY HERE"



category_list_all = {
 'DescribableTextures': dtd, 'OxfordFlowers': flowers, 'SUN397': sun_397,
    'EuroSAT': eurosat,
    'OxfordPets': oxford_pets,
    'Food101': food101,
    'UCF101': ucf_101,
    'ImageNetR': imagenet_rendition

}


vowel_list = ['A', 'E', 'I', 'O', 'U']

Path(f"generic").mkdir(parents=True, exist_ok=True)

for k, v in category_list_all.items():
    all_json_dict = {}
    all_responses = {}
    print('Generating descriptions for ' + k + ' dataset.')
    json_name_all = f"generic/{k}.json"

    if Path(json_name_all).is_file():
        raise ValueError("File already exists")

    for i, category in enumerate(tqdm(v)):
        if category[0].upper() in vowel_list:
            article = "an"
        else:
            article = "a"

        if '_' in category:
            cat = category.replace('_', ' ')
        else:
            cat = category

        prompts = []
        prompts.append("Describe what " + article + " " + cat + " looks like")
        prompts.append("How can you identify " + article + " " + cat + "?")
        prompts.append("What does " + article + " " + cat + " look like?")
        prompts.append("Describe an image from the internet of " + article + " " + cat)
        prompts.append("A caption of an image of " + article + " " + cat + ":")
        all_result = []
        for curr_prompt in prompts:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=curr_prompt,
                temperature=.99,
                max_tokens=50,
                n=5,
                stop="."
            )

            for r in range(len(response["choices"])):
                result = response["choices"][r]["text"]
                all_result.append(result.replace("\n\n", "") + ".")

        all_responses[category] = all_result

        # if i % 10 == 0:
    with open(json_name_all, 'w') as f:
        json.dump(all_responses, f, indent=4)
