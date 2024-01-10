import json
import os
import argparse

from llm.query_llm import OpenAI_LLM
from vln.utils import load_dataset

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--split', default='dev', type=str)
parser.add_argument('--scenario', default='unseen', type=str)
parser.add_argument('--exp_name', default='5shot', type=str)  # is used to name output file
parser.add_argument('--model_name', default='openai/text-davinci-003', type=str)
parser.add_argument('--api_key', default='', type=str)  # OpenAI API key
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances
parser.add_argument('--max_tokens', default=1024, type=int)  # api parameter
parser.add_argument('--prompt_file', default='5shot.txt', type=str)  # filename in llm/prompts/{dataset_name}/landmarks/
parser.add_argument('--output_dir', default='./outputs', type=str)
opts = parser.parse_args()


# settings
exp_name = opts.exp_name
split = opts.split
scenario = opts.scenario
dataset_name = opts.dataset_name
max_tokens = opts.max_tokens
model = tuple(opts.model_name.split('/'))
prompt_file = opts.prompt_file
num_instances = opts.num_instances
api_key = opts.api_key


def main():
    dataset_dir = os.path.join(opts.datasets_dir, dataset_name + '_' + scenario)
    output_dir = os.path.join(opts.output_dir, dataset_name, 'landmarks', model[1], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'{exp_name}_unfiltered.json')

    llm = OpenAI_LLM(max_tokens=opts.max_tokens,
                     model_name=model[-1],
                     api_key=opts.api_key,
                     cache_name='landmarks',
                     finish_reasons=['stop', 'length'])

    prompt_template = get_prompt_template(dataset_name, prompt_file)

    results = dict()
    results['model'] = model
    results['prompt_template'] = prompt_template
    results['max_tokens'] = max_tokens
    results['instances'] = dict()

    if os.path.isfile(results_file):
        with open(results_file) as f:
            results = json.load(f)
    print('results so far: ', len(results['instances']))

    data = load_dataset(split, dataset_dir)

    if num_instances > 0:
        data = data[:num_instances]

    for i, instance in enumerate(data):

        print('results so far: ', len(results['instances']))
        if dataset_name == 'touchdown':
            instance['id'] = instance['route_id']

        idx = str(instance['id'])
        if idx in results['instances']:
            print('skip')
            continue

        print(i, 'number of instances processed')
        print('idx', instance['id'])
        try:
            result = get_landmarks(instance, llm, prompt_template)
        except KeyboardInterrupt:
            llm.save_cache()
            exit()
        idx = str(result['id'])
        results['instances'][idx] = result

    print(len(results['instances']))

    llm.save_cache()
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=1)
    print('queried_tokens', llm.queried_tokens)
    print('wrote results to: ', results_file)


def get_prompt_template(dataset_name, prompt_file):
    with open(os.path.join('llm', 'prompts', dataset_name, 'landmarks', prompt_file)) as f:
        prompt_template = ''
        for line in f:
            prompt_template += line
    return prompt_template


def get_landmarks(instance, llm, prompt_template):
    if dataset_name == 'map2seq':
        idx = instance["id"]
        instance_idx = 'map2seq_' + str(instance["id"])
        instructions_id = instance['instructions_id']
    else:
        idx = instance["route_id"]
        instance_idx = 'touchdown_' + str(instance["route_id"])
        instructions_id = instance["route_id"]

    instructions = instance['navigation_text']
    prompt = prompt_template.format(instructions)

    print('instance_idx', instance_idx)
    print('instructions', instructions)

    sequence = llm.get_sequence(prompt, instance_idx)
    output = sequence.split(prompt)[1]
    unfiltered = get_unfiltered(output)
    print('output', output)
    print('unfiltered', unfiltered)

    result = dict(id=idx,
                  instructions_id=instructions_id,
                  instructions=instructions,
                  unfiltered=unfiltered)
    print('queried_tokens', llm.queried_tokens)
    print('')
    return result


def get_unfiltered(sequence):
    sequence = sequence.split('\n')[1:]
    elements = list()

    if len(sequence) == 1 and sequence[0] == None:
        return [None]

    expected_num = 1
    for element in sequence:
        num = element.split('.')[0]
        try:
            num = int(num)
        except ValueError:
            pass
        if num == expected_num:
            element = ' '.join(element.split(str(num) + '.')[1:]).strip()
            elements.append(element)
            expected_num += 1
    return elements


if __name__ == '__main__':
    main()
