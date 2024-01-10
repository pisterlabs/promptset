import os
import json

import openai
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

openai.api_key = os.getenv('OPENAI_KEY')

data = pd.read_csv('data/data.csv')

new_df = pd.DataFrame({'Interview AI': data['Text'].iloc[::2].values, 'Human': data['Text'].iloc[1::2].values})
print(new_df.head(5))

output = []
for index, row in new_df.iterrows():
    completion = ''
    line = {'prompt': row['Human'], 'completion': row['Interview AI']}

    output.append(line)


with open('data/data.jsonl', 'w') as outfile:
    for i in output:
        json.dump(i, outfile)
        outfile.write('\n')

os.system("openai tools fine_tunes.prepare_data -f 'data/data.jsonl' ")

os.system("openai api fine_tunes.create -t 'data/data_prepared.jsonl' -m davinci ")

# In case training is interrupted, to resume training use
# os.system("openai api fine_tunes.follow -i ft-jl6Ofsj1vRHTuJTlxd5gI59v ")
