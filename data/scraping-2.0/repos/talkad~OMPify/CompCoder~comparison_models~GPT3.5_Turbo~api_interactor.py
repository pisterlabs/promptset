from openai import OpenAI
from prettytable import PrettyTable
from tqdm import tqdm
import json

start_idx = 1501

with open("openai.key", "r") as f:
    key = f.read()

client = OpenAI(api_key=key)

prompt = """Generate the optimal OpenMP pragma for the provided code:
{}
"""

test_file = '/home/talkad/LIGHTBITS_SHARE/OMP_Dataset/cpu/source/test.jsonl'
with open(test_file, 'r') as f, open('output.log', 'w') as out:
    samples = f.readlines()

    for idx, line in tqdm(enumerate(samples[start_idx:])):
        sample = json.loads(line)

        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt.format(sample["code"])}
            ]
            )

            output = {'code': sample["code"],
                    'label': sample["pragma"],
                    'prediction': response.choices[0].message.content}

            out.write(json.dumps(output) + '\n')
        except Exception as e:
            print(f'failed at sample {start_idx+idx}')


# 1227, 1377, 1501, 1567, 1993, 2201, 2323, 2522 samples failed to exceeded length
