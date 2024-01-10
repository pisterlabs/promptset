# %%
import json
import openai
import argparse


# %%
with open("./config.json") as f:
    config_data = json.loads(f.read())

OPENAI_KEY = config_data['OPENAI_KEY']
openai.api_key = OPENAI_KEY
print(OPENAI_KEY)


# %%
def openai_response(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt["refined_prompt"]
        + "\nGenerate a RegEx for this description:\n\n",
        temperature=0.8,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=10,
    )
    print(response)
    prompt['text_davinci_003_output'] = response
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
    item = openai_response(item)
    new_data.append(item)
    break


# %%

if prompt_type == 'refined':
    with open("./Output/Text_DaVinci_Refined_Output.json", "w") as f:
        json.dump(new_data, f, indent=4)

else:
    with open("./Output/Text_DaVinci_Raw_Output.json", "w") as f:
        json.dump(new_data, f, indent=4)


