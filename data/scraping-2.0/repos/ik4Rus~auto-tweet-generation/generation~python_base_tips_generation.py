# Prepare env
import os

import ipdb

import openai
import re
import json

# Parameters
MAX_TOKENS = 500
openai.api_key = os.getenv("OPENAI_KEY")

with open('../data/areas_python_tips.json') as f:
    python_tip_areas = json.load(f)

final_tips_list = []

for area in python_tip_areas:
    print(f"Generating tips for '{area.get('area')}'")
    area_prompt = \
        f"Generate a list of {area.get('count')} advanced Python tips and tricks for the area '{area.get('area')}'. " \
        f"Each tip or trick should be a short sentence, advanced, very specific and they need to be mutually exclusive."
    cnt = 0
    while cnt < 3:
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=area_prompt,
                temperature=0.7,
                max_tokens=MAX_TOKENS,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            tips = response.choices[0].text
            break
        except Exception as e:
            cnt += 1
            print(f"Error ({e}); retrying ({cnt})")
            continue

    # Split the tips into a list
    tips_list = tips.split('\n')
    tips_list = [re.sub(r'^[0-9]*\.*', '', tip).strip() for tip in tips_list if tip != '']

    final_tips_list = final_tips_list + [{"tip": tip, "area": area.get('area')} for tip in tips_list]

with open('../data/python_tips_base.json', 'w+') as f:
    json.dump(final_tips_list, f)

print(f"Generated {len(final_tips_list)} tips")
