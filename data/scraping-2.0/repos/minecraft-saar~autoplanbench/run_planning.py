import os
import openai
from ast import literal_eval
from argparse import ArgumentParser
from llm_planning.planning_set_up import play_games, get_game_class, set_up_configurations
from set_env import set_env_vars

set_env_vars()
openai.api_key = os.environ['OPENAI_API_KEY']


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to the planning config file')
    parser.add_argument('--few-shot-id', required=True,
                        help='ID of the few-shot example to use. Will be selected from the few-shot example directory of the specific approach.')
    args = parser.parse_args()

    config_file = args.config

    few_shot_id = literal_eval(args.few_shot_id) if isinstance(args.few_shot_id, str) else args.few_shot_id

    config, few_shot_path = set_up_configurations(args.config, few_shot_id)
    encoding_type = config.get('encoding_type', 'automatic')
    thoughts = config['thoughts']
    planbench = True if encoding_type == 'planbench' else False

    game_class = get_game_class(thoughts=thoughts, planbench=planbench)
    play_games(config=config, few_shot_path=few_shot_path, game_class=game_class)

