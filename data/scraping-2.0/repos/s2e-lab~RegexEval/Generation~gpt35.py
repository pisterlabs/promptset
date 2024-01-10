# %%
import json
import openai

import argparse


# %%
with open("./config.json") as f:
    config_data = json.loads(f.read())

OPENAI_KEY = config_data['OPENAI_KEY']
openai.api_key = OPENAI_KEY


# %%
def gpt35_response(prompt, prompt_type):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "user",
                    "content": prompt["refined_prompt"] if prompt_type == 'refined' else prompt["raw_prompt"]
                    + "\nGenerate a RegEx for this description:\n\n",
                }
            ],
            temperature=0.8,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=10,
        )
        prompt['gpt3.5_output'] = response
        return prompt
    except Exception as e:
        print(e)
        prompt['gpt3.5_output'] = e
        return prompt



# %%
    
parser = argparse.ArgumentParser(description='GPT3.5')
parser.add_argument('--prompt_type', type=str, default='raw', help='Enter prompt type: raw, refined. Default is raw.')
args = parser.parse_args()

prompt_type = args.prompt_type

print(prompt_type)

if prompt_type not in ['raw', 'refined']:
    raise ValueError('Invalid prompt type. Please enter raw or refined.')

# %%
with open('../DatasetCollection/RegexEval.json') as f:
    data = json.loads(f.read())

print(len(data))

# %%
new_data = []
for item in data:
    item = gpt35_response(item, prompt_type)
    new_data.append(item)


# %%
if prompt_type == 'refined':
    with open('./Output/GPT3.5_Refined_Output.json', "w") as f:
        json.dump(new_data, f, indent=4)
else:
    with open('./Output/GPT3.5_Raw_Output.json', "w") as f:
        json.dump(new_data, f, indent=4)
