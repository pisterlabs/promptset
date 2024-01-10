import json
import os

import openai
import pandas as pd
from tqdm import tqdm

from gpt.gpt_constants import OPEN_AI_MODEL_NAME

openai.api_key = os.environ.get('OPENAI_API_KEY')

gpt_inferences = str()
premise_prompts = pd.read_json(path_or_buf='lewis_caroll_syllogisms_all_no_univ.jsonl', lines=True)
for premise_prompt_row in tqdm(premise_prompts.iterrows()):
    premise_concat = premise_prompt_row[1].tolist()[0]
    premises = premise_concat.split(sep='|')
    premise_prompt = '\n'.join(premises)
    try:
        response = openai.Completion.create(
            engine=OPEN_AI_MODEL_NAME,
            prompt=premise_prompt,
            temperature=0,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0,
            presence_penalty=0)
    except Exception as exception:
        print(exception)
    else:
        response_text = response['choices'][0]['text'].strip()
        gpt_inferences += json.dumps({'premises': premise_prompt, 'conclusion': response_text})
        gpt_inferences += '\n'

gpt_fine_tune_file = open(file='gpt_text_davinci_002_inferences.jsonl', mode='w', encoding='UTF-8')
gpt_fine_tune_file.write(gpt_inferences)
gpt_fine_tune_file.close()