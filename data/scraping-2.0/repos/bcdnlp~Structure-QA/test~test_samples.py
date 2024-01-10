import json
import openai
import yaml
from tqdm import tqdm

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)

with open(config['openai_org_id'], 'r') as f:
    openai.organization = f.readline().strip()

with open(config['openai_key'], 'r') as f:
    openai.api_key = f.readline().strip()

with open(config['hotpotqa_dev_path'], 'r') as f:
    all_data = json.load(f)

for idx in tqdm(range(len(all_data))):
    data = all_data[idx]
    documents = '\n'.join(['Documents:', data['documents']])
    input_text_few_shot = documents
    input_text_few_shot_cot = documents
    input_text_graph_few_shot = documents
    input_text_graph_few_shot_cot = documents

    graph = '\n'.join(['Graph:', data['graph']])
    input_text_graph_few_shot = '\n\n'.join([input_text_graph_few_shot, graph])
    input_text_graph_few_shot_cot = '\n\n'.join([input_text_graph_few_shot_cot, graph])

    question = '\n'.join(['Question:', data['question']])
    input_text_few_shot = '\n\n'.join([input_text_few_shot, question])
    input_text_few_shot_cot = '\n\n'.join([input_text_few_shot_cot, question])
    input_text_graph_few_shot = '\n\n'.join([input_text_graph_few_shot, question])
    input_text_graph_few_shot_cot = '\n\n'.join([input_text_graph_few_shot_cot, question])

    answer = '\n'.join(['Answer:', data['answer']])
    input_text_few_shot = '\n\n'.join([input_text_few_shot, answer])
    input_text_graph_few_shot = '\n\n'.join([input_text_graph_few_shot, answer])

    answer_CoT = '\n'.join(['Answer:', data['answer_CoT']])
    input_text_few_shot_cot = '\n\n'.join([input_text_few_shot_cot, answer_CoT])
    input_text_graph_few_shot_cot = '\n\n'.join([input_text_graph_few_shot_cot, answer_CoT])
    
    input_text_few_shot += '\n\n'
    input_text_few_shot_cot += '\n\n'
    input_text_graph_few_shot += '\n\n'
    input_text_graph_few_shot_cot += '\n\n'

    input_texts = {'prompt_few_shot': input_text_few_shot, 
                   'prompt_few_shot_cot': input_text_few_shot_cot,
                   'prompt_graph_few_shot': input_text_graph_few_shot,
                   'prompt_graph_few_shot_cot': input_text_graph_few_shot_cot}

    for key, input_text in input_texts.items():
        message = {'role': 'user', 'content': input_text}
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo-0301',
                                                messages=[message],
                                                temperature=0,
                                                max_tokens=300,
                                                frequency_penalty=0,
                                                presence_penalty=0,
                                                stop=['\n'],
                                                n=1,
                                                )
        usage = response['usage']['prompt_tokens']
        all_data[idx][key] = usage

with open('output.json', 'w') as f:
    json.dump(all_data, f)

