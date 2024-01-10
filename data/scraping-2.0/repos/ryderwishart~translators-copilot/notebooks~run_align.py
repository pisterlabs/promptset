import openai
import os 
import json
import random
import time

openai.api_key = os.environ['OPENAI_API_KEY']
openai.organization = os.environ['OPENAI_API_ORG']

MAX_RETRIES = 10

### PARSE ARGS ###

import argparse

# Create the argument parser
parser = argparse.ArgumentParser(description='Script to parse command line arguments.')

parser.add_argument('--run_name', type=str, default='default_run', help='Name of the run')
parser.add_argument('--data_path', type=str, default='/Users/ryderwishart/translators-copilot/data/bible/spapddpt.json', help='Path to the data file')
parser.add_argument('--model', type=str, default='gpt-3.5-turbo-instruct', help='Name of the model')
parser.add_argument('--n', type=int, default=0, help='Number of verses to sample')
parser.add_argument('--ids_file_path', type=str, default=None, help='Path to the txt file containing vref ids')

# Parse the arguments
args = parser.parse_args()

# Print the parsed arguments
print(f"Run Name: {args.run_name}")
print(f"Data Path: {args.data_path}")
print(f"Model: {args.model}")
if args.n == 0:
    print("Number of Verses to Sample: ALL")
else:
    print(f"Number of Verses to Sample: {args.n}")
specified_ids = []
if args.ids_file_path:
    with open(args.ids_file_path, 'r') as f:
        specified_ids = [line.strip() for line in f.readlines()]
    print(f"Specified IDs: {specified_ids}")


### ALIGNMENT FUNCTIONS ###
def generate_broad_greek_alignment_prompt(verse):
    bsb, macula, target = verse['bsb']['content'], verse['macula']['content'], verse['target']['content']
    try:
        return f'''Translation style:
The French translation is  a literal translation trying to stick closely to the Hebrew word order, but there may occasionally be instances where Target phrases differ to produce a more natural translation.

Here is a sentence:
Target: Sur quoi Thomas, appelé Didyme, dit aux autres disciples: Allons aussi, afin de mourir avec lui.
English: Then Thomas called Didymus said to his fellow disciples, “Let us also go, so that we may die with Him.”
Greek: εἶπεν οὖν Θωμᾶς ὁ λεγόμενος Δίδυμος τοῖς συνμαθηταῖς Ἄγωμεν καὶ ἡμεῖς ἵνα ἀποθάνωμεν μετ’ αὐτοῦ.

Here is a phonological, semantic, orthographic alignment of that sentence:
```
[
    {{
        "Target phrase": "Sur quoi",
        "English phrase": "Then",
        "Greek phrase": "οὖν"
    }},
    {{
        "Target phrase": "Thomas,",
        "English phrase": "Thomas",
        "Greek phrase": "Θωμᾶς"
    }},
    {{
        "Target phrase": "appelé",
        "English phrase": "called",
        "Greek phrase": "ὁ λεγόμενος"
    }},
    {{
        "Target phrase": "Didyme,",
        "English phrase": "Didymus",
        "Greek phrase": "Δίδυμος"
    }},
    {{
        "Target phrase": "dit",
        "English phrase": "said",
        "Greek phrase": "εἶπεν"
    }},
    {{
        "Target phrase": "aux autres disciples:",
        "English phrase": "to his fellow disciples,",
        "Greek phrase": "τοῖς συνμαθηταῖς"
    }},
    {{
        "Target phrase": "Allons aussi,",
        "English phrase": "“Let us also go,",
        "Greek phrase": "Ἄγωμεν καὶ ἡμεῖς "
    }},
    {{
        "Target phrase": "afin",
        "English phrase": "so that",
        "Greek phrase": "ἵνα"
    }},
    {{
        "Target phrase": "de mourir",
        "English phrase": "we may die",
        "Greek phrase": "ἀποθάνωμεν"
    }},
    {{
        "Target phrase": "avec lui.",
        "English phrase": "with Him.”",
        "Greek phrase": "μετ’ αὐτοῦ."
    }}
]
```

Please also align the following sentence. Avoid including multiple phrases in a single alignment unit. You may need to break phrases  on commas or other major punctuation, including enclosing quotation marks. But you may also need to break a phrase along conjunctions or other words that typically mark the start of a new phrase. Try to align in a fairly granular manner. Always respond with perfect JSON:

Target: {target}
English: {bsb}
Greek: {macula}
'''
    except Exception as e:
        print('Error on Greek alignment prompt generation.', e)
        return 'ERROR'

def generate_broad_hebrew_alignment_prompt(verse): # FIXME: add a French example, and make it more granular
    bsb, macula, target = verse['bsb']['content'], verse['macula']['content'], verse['target']['content']
    try:
        return f'''Translation style:
The French translation is  a literal translation trying to stick closely to the Hebrew word order, but there may occasionally be instances where Target phrases differ to produce a more natural translation.

Here is a sentence:
French: Puis Dieu dit: Que la terre produise de la verdure, de l’herbe portant de la semence, des arbres fruitiers donnant du fruit selon leur espèce et ayant en eux leur semence sur la terre. Et cela fut ainsi.
English: Then God said, “Let the earth bring forth vegetation: seed-bearing plants and fruit trees, each bearing fruit with seed according to its kind.” And it was so.
Hebrew: וַיֹּ֣אמֶר  אֱלֹהִ֗ים  תַּֽדְשֵׁ֤א  הָאָ֨רֶץ֙  דֶּ֔שֶׁא  עֵ֚שֶׂב  מַזְרִ֣יעַ  זֶ֔רַע  עֵ֣ץ  פְּרִ֞י  עֹ֤שֶׂה  פְּרִי֙  לְמִינ֔וֹ  אֲשֶׁ֥ר  זַרְעוֹ־ ב֖וֹ  עַל־ הָאָ֑רֶץ  וַֽיְהִי־ כֵֽן׃

Here is a phonological, semantic, orthographic alignment of that sentence:
```
[
    {{
        "Target phrase": "de la verdure,",
        "English phrase": "vegetation:",
        "Hebrew phrase": "דֶּ֔שֶׁא"
    }},
    {{
        "Target phrase": "de l’herbe portant de la semence,",
        "English phrase": "seed-bearing plants",
        "Hebrew phrase": "עֵ֚שֶׂב מַזְרִ֣יעַ זֶ֔רַע"
    }},
    {{
        "Target phrase": "des arbres fruitiers",
        "English phrase": "and fruit trees,",
        "Hebrew phrase": "עֵ֣ץ פְּרִ֞י"
    }},
    {{
        "Target phrase": "donnant",
        "English phrase": "each bearing",
        "Hebrew phrase": "עֹ֤שֶׂה"
    }},
    {{
        "Target phrase": "du fruit",
        "English phrase": "fruit",
        "Hebrew phrase": "פְּרִי֙"
    }},
    {{
        "Target phrase": "selon leur espèce",
        "English phrase": "according to its kind.”",
        "Hebrew phrase": "לְמִינ֔וֹ"
    }},
    {{
        "Target phrase": "et ayant en eux leur semence",
        "English phrase": "with seed",
        "Hebrew phrase": "אֲשֶׁ֥ר זַרְעוֹ־ ב֖וֹ"
    }},
    {{
        "Target phrase": "sur la terre.",
        "English phrase": "",
        "Hebrew phrase": "עַל־הָאָ֑רֶץ"
    }},
    {{
        "Target phrase": "Et cela fut ainsi.",
        "English phrase": "And it was so.",
        "Hebrew phrase": "וַֽיְהִי־כֵֽן׃"
    }}
]
```

Please also align the following sentence. Avoid including multiple phrases in a single alignment unit. You may need to break phrases on commas or other major punctuation, including enclosing quotation marks. But you may also need to break a phrase along conjunctions or other words that typically mark the start of a new phrase. Try to align in a fairly granular manner. Always respond with perfect JSON:

Target: {target}
English: {bsb}
Hebrew: {macula}
'''
    except Exception as e:
        return 'ERROR'
    
book_idx = {'GEN': 1, 'EXO': 2, 'LEV': 3, 'NUM': 4, 'DEU': 5, 'JOS': 6, 'JDG': 7, 'RUT': 8, '1SA': 9, '2SA': 10,
 '1KI': 11, '2KI': 12, '1CH': 13, '2CH': 14, 'EZR': 15, 'NEH': 16, 'EST': 17, 'JOB': 18, 'PSA': 19, 'PRO': 20,
 'ECC': 21, 'SNG': 22, 'ISA': 23, 'JER': 24, 'LAM': 25, 'EZK': 26, 'DAN': 27, 'HOS': 28, 'JOL': 29, 'AMO': 30,
 'OBA': 31, 'JON': 32, 'MIC': 33, 'NAM': 34, 'HAB': 35, 'ZEP': 36, 'HAG': 37, 'ZEC': 38, 'MAL': 39, 'MAT': 40,
 'MRK': 41, 'LUK': 42, 'JHN': 43, 'ACT': 44, 'ROM': 45, '1CO': 46, '2CO': 47, 'GAL': 48, 'EPH': 49, 'PHP': 50,
 'COL': 51, '1TH': 52, '2TH': 53, '1TI': 54, '2TI': 55, 'TIT': 56, 'PHM': 57, 'HEB': 58, 'JAS': 59, '1PE': 60,
 '2PE': 61, '1JN': 62, '2JN': 63, '3JN': 64, 'JUD': 65, 'REV': 66}

def generate_broad_alignment_prompt(data_element):
    reference = data_element['vref']
    if book_idx[reference[:3]] < 40:
        return generate_broad_hebrew_alignment_prompt(data_element)
    else:
        return generate_broad_greek_alignment_prompt(data_element)

def align(prompt):
    formatted_prompt = ("You are LangAlignerGPT. Analyze the user-supplied alignment examples below and follow any instructions the user gives. Always respond with perfect JSON.\n"
                        f'{prompt}')


    for i in range(MAX_RETRIES):
        try:
            response = openai.Completion.create(
              model=args.model,
              prompt=formatted_prompt,
              temperature=0.1,
              max_tokens=1200,
            #   api_key = os.environ['OPENAI_API_KEY'],
            )
            return response['choices'][0]['text']
        except (openai.error.APIConnectionError, openai.error.APIError) as e:
            if i < MAX_RETRIES - 1:  # i is zero indexed
                continue
            else:
                return {"error": str(e)}

### CREATE OUTPUT DIR ###
import datetime
import os

output_dir = f'data/alignments/{args.run_name}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
### LOAD DATA ###

if args.n == 0:
    if args.data_path.endswith('.jsonl'):
        with open(args.data_path, 'r') as f:
            json_data = [json.loads(line) for line in f]
    else:
        with open(args.data_path, 'r') as f:
            json_data = json.load(f)
else:
    if args.data_path.endswith('.jsonl'):
        with open(args.data_path, 'r') as f:
            json_data = random.sample([json.loads(line) for line in f], args.n)
    else:
        with open(args.data_path, 'r') as f:
            json_data = random.sample(json.load(f), args.n)

# Post-filter json_data by verses in specified_ids array
if specified_ids:
    json_data = [verse for verse in json_data if verse['vref'] in specified_ids]

### ALIGN ###

output_file_path = f'{output_dir}/alignments_{os.path.basename(args.data_path)}_{args.run_name}_{args.model}_{datetime.datetime.now().strftime("%Y%m%d-%H%M")}.jsonl'

for verse in json_data:
    prompt = generate_broad_alignment_prompt(verse)
    try:
        print('aligning', verse['vref'])
        response = align(prompt)
        output = json.loads(response) # FIXME: use pyjson5 ?
        verse['alignment'] = output
        verse['error'] = 'false'
    except json.JSONDecodeError:
        verse['alignment'] = 'Error: Maximum retries exceeded'
        verse['error'] = 'true'
    with open(output_file_path, 'a') as f:
        f.write(json.dumps(verse, ensure_ascii=False))
        f.write('\n')
