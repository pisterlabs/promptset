from collections import defaultdict
from openai import OpenAI
import os
import re

client = OpenAI(api_key=os.environ['OPENAI'])

def remove_parenthesis(text):
    return re.sub(r'\([^)]*\)', '', text)

def get_prompt_from_tmdb_id(tmdb_id):
    characters_str = ','.join([remove_parenthesis(x['character']) for x in tmdb_id2credit[tmdb_id]['cast']])
    characters_ids = [str(x['id']) for x in tmdb_id2credit[tmdb_id]['cast']]
    prompt = f"""Estimate the percentage of the script that each character represents from the movie plot.
[Characters]: {characters_str}
[Plot]: {eval_tmdb_id2s[tmdb_id][0]['plot']}\n
Estimate the percentage of the script that each character represents from the movie plot mentioned above. Return the portion of every character in a JSON dictionary, with the character name as key and portion as value."""
    return prompt, {s:i for s,i in zip([remove_parenthesis(x['character']) for x in tmdb_id2credit[tmdb_id]['cast']],characters_ids)}

def infer_chatgpt(prompt):
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      messages=[
        {"role": "user", "content": prompt}
      ],
      response_format={ "type": "json_object" },
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    
    eval_tmdb_id2s = defaultdict(list)
    scripts_in_sets = json.load(open('plot_portion_dataset.json'))
    tmdb_id2credit = json.load(open('../Data/tmdb_resources/tmdb_id2credit_full.json'))
    for s in scripts_in_sets['evaluation']:
        eval_tmdb_id2s[s['tmdb_id']].append(s)
    chatgpt_pred = {}
    for tmdb_id in tqdm(list(eval_tmdb_id2s.keys())):
        prompt, character2id = get_prompt_from_tmdb_id(tmdb_id)
        gen_text = infer_chatgpt(prompt)
        chatgpt_pred[tmdb_id] = {
            'character2id': character2id,
            'gen_text': gen_text,
        }
    json.dump(chatgpt_pred,open('eval_chatgpt_pred.json','w'))
    