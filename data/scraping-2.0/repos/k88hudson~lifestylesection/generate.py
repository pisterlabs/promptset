import os
import openai
import json
import random
import requests

openai.api_key = os.environ["OPENAI_SECRET"]

health_prompts = [
    "is actually good for you",
    "is actually bad for you",
    "is surprisingly unhealthy",
    "is good for you and debunks myths about it",
    "is something you should consider eating",
    "is something you should avoid",
    "impacted people's health in a recent study",
]

def health_prompt(food):
    return f"""Write a headline and lede for a NYT article about
how {food} {random.choice(health_prompts)}, not using those exact words.
The headline and lede must be separated by a % character."""

os.mkdir("output")
os.mkdir("output/img")

with open("foods2.txt", "r") as foods_file, open("output/food_completions.json", "w") as file:
    foods = [f.strip() for f in foods_file.readlines()]

    output = {}
    output["completions"] = []
    food_choices = random.sample(foods, 30)

    for i in range(30):
        food = food_choices[i]
        prompt = health_prompt(food=food)
        try:
            completion = openai.Completion.create(model="text-davinci-003", prompt=prompt, max_tokens=100)
            text = [item.strip() for item in completion.choices[0].text.split("%")]

            img_resp = openai.Image.create(
                prompt=f"product shot of {food}, photography, for a health article, single color background",
                n=1,
                size="512x512"
            )
            image_url = img_resp["data"][0]["url"]
            img_filename = f'{food}_{i}.png'

            # Download image
            try:
                img_data = requests.get(image_url).content
                with open("output/img/" + img_filename, 'wb') as handler:
                    handler.write(img_data)
            except Exception as img_e:
                print(img_e)
                print(img_filename, image_url)

            output["completions"].append({
                "prompt": prompt,
                "text": text,
                "image_url": image_url,
                "image_filename": img_filename
            })
            print(i, text)
        except Exception as e:
            print(e)
    json.dump(output, file, indent=2)

# with open("food_completions.json", "r") as output_file:
#     output = json.load(output_file)
#     for line in output["completions"]:
#         image_url = line["image_url"]
#         image_filename = line["image_filename"]

#         try:
#             img_data = requests.get(image_url).content
#             with open("img/" + image_filename, 'wb') as handler:
#                 handler.write(img_data)
#         except Exception as img_e:
#             print(img_e)
