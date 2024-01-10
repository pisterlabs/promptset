import time
import openai
import pandas as pd
from foi_simple import *
from tqdm import tqdm
from keys import *

openai.api_key = keys['openAI']

# ========================== Parameters ========================================

ITERATIONS = 1
print_stories = True
people_csv = './fake_data/fake_people.csv'
output_csv = './results_simple_multipass_text-davinci-003.csv'

BACKSTORY_START = ""
BACKSTORY_END = ""

# ==============================================================================

def gen_backstory(pid, df):
    person = df.iloc[pid]
    id = person['ids']
    backstory = BACKSTORY_START

    for k in foi_keys:
        df_val = person[k]
        elem_template = fields_of_interest[k]['template']
        elem_map = fields_of_interest[k]['valmap']

        if len(elem_map) == 0:
            newval = str(df_val)
        elif df_val in elem_map:
            newval = elem_map[df_val]
        else:
            newval = str(df_val)

        newval = newval.replace("<1:[RECORD VERBATIM]>:", "")
        backstory += " " + elem_template.replace('XXX', newval)

    if backstory[0] == ' ':
        backstory = backstory[1:]

    return id, backstory

def do_query(prompt, max_tokens=512, engine="text-davinci-003"):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )
    return response

# ==============================================================================

df = pd.read_csv(people_csv)
foi_keys = "age gender ethnicity".split()

ids = []
prompt_ids = []
prompts = []
responses = []
ethnicities = []

for pid in tqdm(range(len(df))):
    for i in range(ITERATIONS):
        prompt_ids.append(i)
        ethnicities.append(df.iloc[pid]['ethnicity'])
        id, prompt = gen_backstory(pid, df)
        prompt += BACKSTORY_END

        if print_stories:
            print("---------------------------------------------------")
            print(prompt)

        done = False
        while not done:
            try:
                response = do_query(prompt, max_tokens=128, engine="text-davinci-003")
                resp_text = response.choices[0]['text']
                if print_stories: print(resp_text)
                ids.append(id)
                prompts.append(prompt)
                responses.append(resp_text)
                done = True
            except:
                print('Exception occurred. Sleeping for 5 seconds')
                time.sleep(5.0)

# ==============================================================================

newdf = pd.DataFrame({'ids': ids, 'pids': prompt_ids, 'prompt': prompts, 'response': responses, 'ethnicity': ethnicities})
newdf.to_csv(output_csv)

print('stories.py finished')
