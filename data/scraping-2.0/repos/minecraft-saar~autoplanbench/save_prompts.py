import os
import openai
import json
from argparse import ArgumentParser
from pathlib import Path
from ast import literal_eval
from llm_planning.planning_set_up import get_prompts, get_game_class, set_up_configurations
from set_env import set_env_vars

set_env_vars()
openai.api_key = os.environ['OPENAI_API_KEY']

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the planning config file')
    parser.add_argument('--few-shot-id', required=True, help='ID of the few-shot example to use. Will be selected from the few-shot example directory of the specific approach.')
    parser.add_argument('--pl-out', required=True, help='Path for the file where the P-LLM prompt gets saved')
    parser.add_argument('--tr-out', required=True, help='Path for the file where the T-LLM prompt gets saved')
    args = parser.parse_args()

    few_shot_id = literal_eval(args.few_shot_id) if isinstance(args.few_shot_id, str) else args.few_shot_id

    config, few_shot_path = set_up_configurations(args.config, few_shot_id)
    encoding_type = config.get('encoding_type', 'automatic')
    thoughts = config['thoughts']
    planbench = True if encoding_type == 'planbench' else False

    game_class = get_game_class(thoughts=thoughts, planbench=planbench)
    plan_prompt, translate_prompt, _ = get_prompts(config=config, few_shot_path=few_shot_path, game_class=game_class)

    output_dir1 = os.path.split(args.pl_out)[0]
    output_dir2 = os.path.split(args.tr_out)[0]
    Path(output_dir1).mkdir(exist_ok=True)
    Path(output_dir2).mkdir(exist_ok=True)

    with open(args.pl_out, 'w') as file:
        file.write(plan_prompt)
    with open(args.tr_out, 'w') as file:
        file.write(translate_prompt)
