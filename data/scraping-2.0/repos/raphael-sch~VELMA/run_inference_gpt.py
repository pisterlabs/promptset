import json
import os
import argparse
import time
import random

import tqdm

from vln.dataset import load_dataset
from vln.prompt_builder import get_navigation_lines

from vln.clip import PanoCLIP
from vln.env import ClipEnv, get_gold_nav

from llm.query_llm import OpenAI_LLM
from vln.evaluate import get_metrics_from_results
from vln.agent import LLMAgent

from functools import partial
import concurrent.futures

parser = argparse.ArgumentParser(description='Define experiment parameters')
parser.add_argument('--datasets_dir', default='./datasets', type=str)
parser.add_argument('--dataset_name', default='map2seq', type=str)
parser.add_argument('--split', default='dev', type=str)
parser.add_argument('--scenario', default='unseen', type=str)
parser.add_argument('--exp_name', default='2shot', type=str)  # is used to name output file
parser.add_argument('--image', default='openclip', choices=['openclip', 'clip', 'none'], type=str)
parser.add_argument('--image_prompt', default='picture of {}', type=str)
parser.add_argument('--image_threshold', default=3.5, type=float)
parser.add_argument('--landmarks_name', default='gpt3_5shot', type=str)
parser.add_argument('--model_name', default='openai/text-davinci-003', type=str) #openai/gpt-4-0314
parser.add_argument('--api_key', default='', type=str)  # OpenAI API key
parser.add_argument('--num_instances', default=-1, type=int)  # -1 for all instances
parser.add_argument('--max_tokens', default=50, type=int)  # api parameter
parser.add_argument('--max_steps', default=55, type=int)  # maximum number of agent steps before run is canceled
parser.add_argument('--prompt_file', default='2shot.txt', type=str)  # filename in llm/ prompts/{dataset_name}/navigation/
parser.add_argument('--clip_cache_dir', default='./features', type=str)
parser.add_argument('--output_dir', default='./outputs', type=str)
parser.add_argument('--n_workers', default=1, type=int)
parser.add_argument('--seed', default=1, type=int)
opts = parser.parse_args()

random.seed(opts.seed)

dataset_name = opts.dataset_name
is_map2seq = dataset_name == 'map2seq'
data_dir = opts.datasets_dir
dataset_dir = os.path.join(data_dir, dataset_name + '_' + opts.scenario)
graph_dir = os.path.join(dataset_dir, 'graph')
landmarks_dir = os.path.join(data_dir, 'landmarks')
landmarks_file = os.path.join(landmarks_dir, dataset_name, f'{opts.landmarks_name}_unfiltered.json')
prompts_dir = os.path.join('llm', 'prompts')

counter = 0


def main():
    output_name = '_'.join([opts.exp_name, opts.image])
    panoCLIP = None
    if opts.image != 'none':
        output_name += '_L' + opts.landmarks_name
        panoCLIP = PanoCLIP(model_name=opts.image, device="cpu", cache_dir=opts.clip_cache_dir)
    env = ClipEnv(graph_dir, panoCLIP, image_threshold=opts.image_threshold, image_prompt=opts.image_prompt)

    model = opts.model_name.split('/')  # ('openai', 'text-davinci-003')
    output_dir = os.path.join(opts.output_dir, dataset_name + '_' + opts.scenario, model[-1])
    os.makedirs(output_dir, exist_ok=True)

    train_instances = load_dataset('train', env, dataset_dir, dataset_name, landmarks_file)

    instances = load_dataset(opts.split, env, dataset_dir, dataset_name, landmarks_file)

    with open(os.path.join(prompts_dir, dataset_name, 'navigation', opts.prompt_file)) as f:
        prompt_template = ''.join(f.readlines())

    llm = OpenAI_LLM(max_tokens=opts.max_tokens,
                     model_name=model[-1],
                     api_key=opts.api_key,
                     cache_name='navigation',
                     finish_reasons=['stop', 'length'])

    results = dict()
    results['opts'] = vars(opts)
    results['prompt_template'] = prompt_template
    results['time'] = int(time.time())
    results['instances'] = dict()

    if opts.num_instances != -1:
        instances = instances[:opts.num_instances]
    print('instances: ', len(instances))

    pbar = tqdm.tqdm(total=len(instances), smoothing=0.1)

    if opts.n_workers > 1:
        args = list()
        for instance in instances:
            icl_shots = random.sample(train_instances, 2)
            args.append((instance, icl_shots, pbar))
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=opts.n_workers) as executor:
                func = partial(process_instance, llm=llm, env=env, prompt_template=prompt_template)
                map_results = list(executor.map(func, args))
        except KeyboardInterrupt:
            llm.save_cache()
            if panoCLIP:
                panoCLIP.save_cache()
            exit()
        except RuntimeError:
            llm.save_cache()
            exit()

        query_count = 0
        for result in map_results:
            results['instances'][result['idx']] = result
            query_count += result['query_count']
            del result['query_count']
        print('')
        print('queried tokens: ', llm.queried_tokens)
        print('total query count: ', query_count)
    else:
        for i, instance in enumerate(instances):
            icl_shots = random.sample(train_instances, 2)

            print(i, 'number of instances processed')
            print('idx', instance['idx'])

            result = process_instance((instance, icl_shots, pbar), llm, env, prompt_template)
            results['instances'][result['idx']] = result

    llm.save_cache()
    if panoCLIP:
        panoCLIP.save_cache()
    correct, tc, spd, kpa, results = get_metrics_from_results(results, env.graph)
    print('')
    print('correct', correct)
    print('tc', tc)
    print('spd', spd)
    print('kpa', kpa)
    print('')

    results_file = os.path.join(output_dir, f'{output_name}_{model[-1]}_{opts.split}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote results to: ', results_file)


def process_instance(args, llm, env, prompt_template):
    instance, icl_shots, pbar = args

    icl_instance_1 = icl_shots[0]
    gold_nav_shot_1 = get_gold_nav(icl_instance_1, env)
    gold_navigation_lines_shot_1, _ = get_navigation_lines(gold_nav_shot_1, env, icl_instance_1['landmarks'], icl_instance_1.get('traffic_flow'))
    icl_instance_2 = icl_shots[1]
    gold_nav_shot_2 = get_gold_nav(icl_instance_2, env)
    gold_navigation_lines_shot_2, _ = get_navigation_lines(gold_nav_shot_2, env, icl_instance_2['landmarks'], icl_instance_2.get('traffic_flow'))

    prompt_template = prompt_template.format(icl_instance_1['navigation_text'],
                                             '\n'.join(gold_navigation_lines_shot_1),
                                             icl_instance_2['navigation_text'],
                                             '\n'.join(gold_navigation_lines_shot_2),
                                             '{}')


    # main computation
    agent = LLMAgent(llm, env, instance, prompt_template)
    nav, navigation_lines, is_actions, query_count = agent.run(opts.max_steps)

    gold_nav = get_gold_nav(instance, env)
    gold_navigation_lines, gold_is_actions = get_navigation_lines(gold_nav, env, agent.landmarks, agent.traffic_flow)

    global counter
    counter += 1
    pbar.update()

    print('instance id', instance["id"])
    print('result:')
    print(instance['navigation_text'])
    print(instance['landmarks'])
    print('\n'.join(navigation_lines))
    print('actions', nav.actions)
    print('query_count', query_count)
    print('processed instances', counter)

    result = dict(idx=instance['idx'],
                  navigation_text=instance['navigation_text'],
                  start_heading=instance['start_heading'],
                  gold_actions=gold_nav.actions,
                  gold_states=gold_nav.states,
                  gold_pano_path=instance['route_panoids'],
                  gold_navigation_lines=gold_navigation_lines,
                  gold_is_actions=gold_is_actions,
                  agent_actions=nav.actions,
                  agent_states=nav.states,
                  agent_pano_path=nav.pano_path,
                  agent_navigation_lines=navigation_lines,
                  agent_is_actions=is_actions,
                  query_count=query_count)
    return result


if __name__ == '__main__':
    main()
